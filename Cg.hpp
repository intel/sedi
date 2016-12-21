#ifndef CG_H
#define CG_H

#include <cassert>
#include <cmath>
#include <cstdio>

#include "DomainDecomposition.hpp"
#include "Stencil.hpp"
#include "DistrAsyncStencil.hpp"
#include "DistrAsyncDotProds.hpp"
#include "CommHalo.hpp"
#include "Grid3d.hpp"
#include "SliceOps.hpp"
#include "Slice3d.hpp"

#include "Timers.hpp"

template <typename Dtype>
class Cg {
	public:
	DomainDecomposition & dd;

	// Temp vectors for CG
	Grid3d<Dtype> r; // Residual vector
	Grid3d<Dtype> s, u;

	// These vectors need ghost because the stencil gets applied to them
	Grid3d<Dtype> p;

	typedef enum {
		CG_SOLUTION_FOUND = 0,
		CG_REACHED_MAXITER = 1
	} result_t;

	typedef enum {
		CG_ASYNCHRONOUS = 0,
		CG_SYNCHRONOUS_STENCIL = 1 << 0,
		CG_SYNCHRONOUS_DOTPROD = 1 << 1
	} mode_t;

	// Whether to do the dotprods and stencils in a synchronous way
	const bool sync_dotprod, sync_stencil;

	Timers * const timers;
	SliceOps<Dtype> slice_ops;

	Cg(DomainDecomposition & dd_, long nghost, int mode, Timers *timers_ = 00) :
		dd(dd_),
		r(dd, 0, nghost), s(dd, 0, nghost), u(dd, 0, nghost), 
		// Vectors with ghosts
		p(dd, nghost, nghost),
		sync_dotprod((mode & CG_SYNCHRONOUS_DOTPROD) != 0),
		sync_stencil((mode & CG_SYNCHRONOUS_STENCIL) != 0),
		timers(timers_), slice_ops(timers)
	{
		// Make sure ghosts are set to 0, otherwise we will be picking up
		// affine terms in the stencil. Some of these ghosts will be updated by
		// MPI comms, but others correspond to global domain boundaries and
		// will only be read from now on (which is why they have to be set to 0).
		// XXX: this may need refactoring/adapting to other operators
		p.fill_ghosts(0.0);

		if (dd.myrank == 0) {
			printf("Initialized Cg solver, sync_dotprod=%d, sync_stencil=%d\n",
					sync_dotprod, sync_stencil);
		}
	}

	result_t solve(
			// Inputs
			Stencil<Dtype> & A, // A: Stencil operator
			CommHalo<Dtype> & comm_halo, // Halo exchange buffers
			Grid3d<Dtype> & x, // Solution vector (first guess & output)
			Grid3d<Dtype> & b, // b: right hand side
			Grid3d<Dtype> & iM, // iM: inverse diagonal preconditioner in vector form
			int maxiter, // maximum number of iterations
			Dtype tol, // tolerance for convergence
			// Outputs
			Dtype *final_err, // final error
			int *final_iter // final iteration count
			) {

		// Setup asynchronous dot products
		DistrAsyncDotProds<Dtype> dp_b_b = DistrAsyncDotProds<Dtype>::single_dotprod(b, b, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_s_p = DistrAsyncDotProds<Dtype>::single_dotprod(s, p, sync_dotprod, timers);

		DistrAsyncDotProds<Dtype> dp_r_r = DistrAsyncDotProds<Dtype>::single_dotprod(r, r, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_r_u = DistrAsyncDotProds<Dtype>::single_dotprod(r, u, sync_dotprod, timers);

		Grid3d<Dtype> * rr[] = {&r, &r};
		Grid3d<Dtype> * ur[] = {&u, &r};
		DistrAsyncDotProds<Dtype> dp_ru_rr = DistrAsyncDotProds<Dtype>::multi_dotprod(rr, ur, 2, sync_dotprod, timers);

		// Setup asynchronous stencil operations on vectors
		// NOTE: all these operators share the same CommHalo, so they can't be
		// overlapped. To overlap them safely, we'd need to create one CommHalo
		// object for each so that they don't share buffers
		DistrAsyncStencil<Dtype> stencil_x_r(A, x, r, comm_halo, sync_stencil, timers);
		DistrAsyncStencil<Dtype> stencil_p_s(A, p, s, comm_halo, sync_stencil, timers);

		// Setup slices
		Slice3d<Dtype> x_ = x.rank_cells;
		Slice3d<Dtype> b_ = b.rank_cells;
		Slice3d<Dtype> iM_ = iM.rank_cells;
		Slice3d<Dtype> r_ = r.rank_cells;
		Slice3d<Dtype> s_ = s.rank_cells;
		Slice3d<Dtype> u_ = u.rank_cells;
		Slice3d<Dtype> p_ = p.rank_cells;

		// Compute the square norm of b (i.e. <b,b>)
		dp_b_b.initiate();
		Dtype b_norm2;
		dp_b_b.finalize(&b_norm2);
		if (b_norm2 == 0.0) b_norm2 = 1.0;

		// Compute the residual:
		// 1. apply operator: r <- A*x
		stencil_x_r.initiate();
		stencil_x_r.compute_local();
		stencil_x_r.finalize();
		// 2. r <- b - A*x = b - r
		slice_ops.combine2(1.0, b_, -1.0, r_, r_);
		
		// Compute <r,r>
		dp_r_r.initiate();
		Dtype r_norm2;
		dp_r_r.finalize(&r_norm2);
		Dtype err = sqrt(r_norm2/b_norm2);
		if (err < tol) {
			*final_iter = 0;
			*final_err = err;
			return CG_SOLUTION_FOUND;
		}

		Dtype alpha = 0.0;
		Dtype beta = 0.0;
		Dtype s_dot_p = 0.0;
		Dtype r_dot_u = 0.0;
		Dtype r_dot_u_new = 0.0;

		// Initialize u
		// u <- r ./ M
		slice_ops.multiply(r_, iM_, u_);

		// Initialize p
		slice_ops.copy(u_, p_);
		
		int iter;
		for (iter=1; iter<maxiter+1; iter++) {
			// s <- A*p
			stencil_p_s.initiate();
			stencil_p_s.compute_local();
			stencil_p_s.finalize();

			// alpha <- (r,u) / (s,p)
			if (iter == 1) {
				dp_r_u.initiate();
				dp_r_u.finalize(&r_dot_u);
			} else {
				// (r, u) was already computed at end of previous iter
				r_dot_u = r_dot_u_new;
			}
			dp_s_p.initiate();
			dp_s_p.finalize(&s_dot_p);
			alpha = r_dot_u / s_dot_p;

			// x <- x + alpha*p
			slice_ops.combine2(1.0, x_, alpha, p_, x_);

			// r <- r - alpha*s
			slice_ops.combine2(1.0, r_, -alpha, s_, r_);

			// u <- r ./ M
			slice_ops.multiply(r_, iM_, u_);

			// New (r, u), (r, r)
			Dtype ru_rr[2];
			dp_ru_rr.initiate();
			dp_ru_rr.finalize(ru_rr);
			r_dot_u_new = ru_rr[0];
			r_norm2 = ru_rr[1];

			// Update beta
			beta = r_dot_u_new / r_dot_u;

			// p <- u + beta*p
			slice_ops.combine2(1.0, u_, beta, p_, p_);

			// Check convergence
			err = sqrt(r_norm2/b_norm2);
			if (dd.myrank == 0) printf("Cg: after iter=%d, err=%.10e\n", iter, err);
			if (err <= tol) break;
		}

		// Now check the convergence status
		if (err <= tol) {
			*final_iter = iter;
			*final_err = err;
			return CG_SOLUTION_FOUND;
		}

		assert(iter == maxiter+1);
		*final_iter = maxiter;
		*final_err = err;
		return CG_REACHED_MAXITER;
	}
	
};

#endif
