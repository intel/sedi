#include <mpi.h>
#include <cstdio>
#include <math.h>

#ifdef WITH_LINUX_EXTENSIONS
#include <fenv.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include "DomainDecomposition.hpp"
#include "Grid3d.hpp"
#include "SliceOps.hpp"
#include "CommHalo.hpp"
#include "Stencil.hpp"
#include "DistrAsyncStencil.hpp"
#include "PipelinedCg.hpp"
#include "Cg.hpp"
#include "BiCgStab.hpp"
#include "CmdLineParser.hpp"
#include "DistrRngStd.hpp"
#include "Timers.hpp"

#ifdef WITH_HDF5
#include "ValidateSolver.hpp"
#include "ValidateEstimator.hpp"
#endif

typedef double real_t;

void collect_and_report_mem_usage(int myrank, int nranks) {
	MPI_Barrier(MPI_COMM_WORLD);
	double rss_mb;

#ifdef WITH_LINUX_EXTENSIONS
	struct rusage usage;
	int err = getrusage(RUSAGE_SELF, &usage);
	assert(err == 0);
	rss_mb = usage.ru_maxrss / 1024.0; // ru_maxrss in kilobytes
#else
	rss_mb = 0.0;
#endif

	double rss_tot, rss_avg, rss_min, rss_max;
	rss_tot = rss_avg = rss_min = rss_max = 0.0;
	MPI_Allreduce(&rss_mb, &rss_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&rss_mb, &rss_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&rss_mb, &rss_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	rss_avg = rss_tot / nranks;

	if (myrank == 0) {
		printf("Resident set size across ranks [MB]: avg=%.3e, tot=%.3e, min=%.3e, max=%.3e\n",
				rss_avg, rss_tot, rss_min, rss_max);
	}
}

int main(int argc, char * argv[]) {

	MPI_Init(&argc, &argv);

	// Parse command line arguments
	CmdLineParser parser(argc, argv);

#ifdef WITH_LINUX_EXTENSIONS
	if (parser.dont_enable_fp_exceptions == false) {
		// Enable FP exceptions to catch NaNs, Infs, divide by zero
		feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);
	}
#endif

	Timers timers;

	// Define stencil and domain decomp
	Stencil<real_t> stencil(6.0, 1.0, 1.0);
	const long nghost = 1;
	const long ninner = nghost;
	DomainDecomposition dd(
			parser.ng1, parser.ng2, parser.ng3,
			parser.np1, parser.np2, parser.np3);

	const real_t tolerance = parser.tolerance; // Convergence tol for solver
	const int    maxiter   = parser.maxiter;   // Max iterations for solver
	const int    samples   = parser.samples;   // number of probing vectors

	// Halos for comms
	CommHalo<real_t> comm_halo(dd, nghost, &timers);

	// Setup vectors
	// Note that only x_k has ghosts, because we don't need to apply the
	// stencil to any other vector (except within the solver, but the solver
	// class takes care of its own temp storage)

	Grid3d<real_t> x_k(dd, nghost, ninner); x_k.all_cells.fill(0.0); // The unknown vector
	// Note that the above line also fills x_k's ghost cells with 0, as req for the stencil

	Grid3d<real_t> v_k(dd, 0     , ninner); v_k.all_cells.fill(0.0); // The random RHS
	Grid3d<real_t> t_k(dd, 0     , ninner); t_k.all_cells.fill(0.0);
	Grid3d<real_t> q_k(dd, 0     , ninner); q_k.all_cells.fill(0.0);
	Grid3d<real_t> D  (dd, 0     , ninner); q_k.all_cells.fill(0.0); // The resulting estimator

	// Setup preconditioner
	Grid3d<real_t> precond(dd, 0, ninner);
	precond.all_cells.fill(1.0/stencil.a); // Inv diagonal precon. based on stencil diagonal term

	// Setup RNG
	DistrRngStd<real_t> rng(dd);

	// Setup validation helpers
#ifdef WITH_HDF5
	ValidateSolver<real_t> * validate_solver = 00;
	ValidateEstimator<real_t> * validate_estimator = 00;

	if (parser.validate_solver_h5_file != 00)
		validate_solver = new ValidateSolver<real_t>(dd, parser.validate_solver_h5_file);

	if (parser.validate_estimator_h5_file != 00)
		validate_estimator = new ValidateEstimator<real_t>(dd, parser.validate_estimator_h5_file);
#else
	// No HDF5
	if ((parser.validate_solver_h5_file != 00) || (parser.validate_estimator_h5_file != 00)) {
		if (dd.myrank == 0) printf("WARNING: validation mode selected, but compiled without HDF5 support: disabling!\n");
		parser.validate_solver_h5_file = 00;
		parser.validate_estimator_h5_file = 00;
	}
#endif

	if ((parser.validate_solver_h5_file != 00) && (parser.validate_estimator_h5_file != 00)) {
		if (dd.myrank == 0) printf("WARNING: cannot enable both solver and estimator validation mode: disabling both!\n");
		parser.validate_solver_h5_file = 00;
		parser.validate_estimator_h5_file = 00;
	}

	// Setup solver
#if   SOLVER == 1
	typedef BiCgStab<real_t> solver_t;
#elif SOLVER == 2
	typedef Cg<real_t> solver_t;
#elif SOLVER == 3
	typedef PipelinedCg<real_t> solver_t;
#else
#error "Invalid solver specified in SOLVER"
#endif

	solver_t solver(dd, nghost, parser.solver_mode, &timers);
	solver_t::result_t solver_outcome;
	int final_iter;
	real_t final_err;

	SliceOps<real_t> slice_ops(&timers);

	// Synchronize & start timers
	MPI_Barrier(MPI_COMM_WORLD);
	timers.start(Timers::TIMER_GLOBAL);

	// Start loop on samples
	int maxiter_over_samples = 0;
	int miniter_over_samples = maxiter+1;
	for (int k=0; k<samples; k++) {

		if (dd.myrank == 0) printf("Solving for sample %d/%d...\n", k+1, samples);

		if (validate_solver == 00) {
			// Fill v_k with random +/- 1
			timers.start(Timers::TIMER_COMPUTE_RNG);
			rng.fill_plusminusone(v_k.rank_cells);
			timers.stop(Timers::TIMER_COMPUTE_RNG);
		} else {
#ifdef WITH_HDF5
			// Fill v_k with reference RHS
			validate_solver->fill_with_ref_rhs(v_k.rank_cells);
#endif
		}

		// Solve stencil(x_k) = v_k
		x_k.all_cells.fill(0.0); // Use 0 as first guess
		solver_outcome = solver.solve(stencil, comm_halo,
				x_k, v_k, precond,
				maxiter, tolerance,
				&final_err, &final_iter);

		if (!parser.ignore_solver_errors && solver_outcome != 0) {
			fprintf(stderr, "Incomplete convergence in solver, errcode=%d\n", solver_outcome);
			MPI_Abort(MPI_COMM_WORLD, 100+solver_outcome);
		}

		if (final_iter > maxiter_over_samples) maxiter_over_samples = final_iter;
		if (final_iter < miniter_over_samples) miniter_over_samples = final_iter;

#ifdef WITH_HDF5
		if (validate_solver != 00) validate_solver->compare_to_ref_solution(x_k.rank_cells);
#endif

		// Update t_k and q_k
		// t_k <- t_k + x_k .* v_k
		slice_ops.accumulate_product(x_k.rank_cells, v_k.rank_cells, t_k.rank_cells);

		// q_k <- q_k + v_k .* v_k
		slice_ops.accumulate_product(v_k.rank_cells, v_k.rank_cells, q_k.rank_cells);

#ifdef WITH_HDF5
		if (validate_estimator != 00) {
			// Update D at this stage (only for validation)
			slice_ops.divide(t_k.rank_cells, q_k.rank_cells, D.rank_cells);
			validate_estimator->compare_to_ref_inv_diag(D.rank_cells);
		}
#endif

	}

	// Synchronize & stop timers
	MPI_Barrier(MPI_COMM_WORLD);
	timers.stop(Timers::TIMER_GLOBAL);

	// Stochastic estimator for diag(inv(stencil))
	// D <- t_k ./ q_k
	slice_ops.divide(t_k.rank_cells, q_k.rank_cells, D.rank_cells);

	if (dd.myrank == 0) {
		printf("Total solve time: %.4e seconds, iteration count min=%d max=%d\n",
			timers.total_time[Timers::TIMER_GLOBAL], miniter_over_samples, maxiter_over_samples);
		timers.print_timers();
	}

	collect_and_report_mem_usage(dd.myrank, dd.np);

#ifdef WITH_HDF5
	if (validate_solver != 00) delete validate_solver;
	if (validate_estimator != 00) delete validate_estimator;
#endif

	MPI_Finalize();
}
