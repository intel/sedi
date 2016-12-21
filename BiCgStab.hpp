#ifndef BI_CG_STAB_H
#define BI_CG_STAB_H

#include <cassert>
#include <cmath>
#include <cstdio>

#include "DomainDecomposition.hpp"
#include "Stencil.hpp"
#include "DistrAsyncStencil.hpp"
#include "DistrAsyncDotProds.hpp"
#include "CommHalo.hpp"
#include "Grid3d.hpp"
#include "Slice3d.hpp"
#include "SliceOps.hpp"

#include "Timers.hpp"

template <typename Dtype>
class BiCgStab {
	public:
	DomainDecomposition & dd;

	// Temp vectors for BiCGStab
	Grid3d<Dtype> r; // Residual vector
	Grid3d<Dtype> r_hat;
	Grid3d<Dtype> p, s, v, t;

	// These vectors need ghost because the stencil gets applied to them
	Grid3d<Dtype> p_hat, s_hat;

	typedef enum {
		BICGSTAB_SOLUTION_FOUND = 0,
		BICGSTAB_REACHED_MAXITER = 1,
		BICGSTAB_BREAKDOWN_RHO = 2,
		BICGSTAB_BREAKDOWN_OMEGA = 3
	} result_t;

	typedef enum {
		BICGSTAB_ASYNCHRONOUS = 0,
		BICGSTAB_SYNCHRONOUS_STENCIL = 1 << 0,
		BICGSTAB_SYNCHRONOUS_DOTPROD = 1 << 1
	} mode_t;

	// Whether to do the dotprods and stencils in a synchronous way
	const bool sync_dotprod, sync_stencil;

	Timers * const timers;
	SliceOps<Dtype> slice_ops;

	BiCgStab(DomainDecomposition & dd_, long nghost, int mode, Timers *timers_ = 00) :
		dd(dd_),
		r(dd, 0, nghost), r_hat(dd, 0, nghost),
		p(dd, 0, nghost), s(dd, 0, nghost),
		v(dd, 0, nghost), t(dd, 0, nghost),
		// Vectors with ghosts
		p_hat(dd, nghost, nghost),
		s_hat(dd, nghost, nghost),
		sync_dotprod((mode & BICGSTAB_SYNCHRONOUS_DOTPROD) != 0),
		sync_stencil((mode & BICGSTAB_SYNCHRONOUS_STENCIL) != 0),
		timers(timers_), slice_ops(timers)
	{
		// Make sure ghosts are set to 0, otherwise we will be picking up
		// affine terms in the stencil. Some of these ghosts will be updated by
		// MPI comms, but others correspond to global domain boundaries and
		// will only be read from now on (which is why they have to be set to 0).
		// XXX: this may need refactoring/adapting to other operators
		p_hat.fill_ghosts(0.0);
		s_hat.fill_ghosts(0.0);

		if (dd.myrank == 0) {
			printf("Initialized BiCgStab solver, sync_dotprod=%d, sync_stencil=%d\n",
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
		DistrAsyncDotProds<Dtype> dp_b_b  = DistrAsyncDotProds<Dtype>::single_dotprod(b, b, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_r_r  = DistrAsyncDotProds<Dtype>::single_dotprod(r, r, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_r_rh = DistrAsyncDotProds<Dtype>::single_dotprod(r, r_hat, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_v_rh = DistrAsyncDotProds<Dtype>::single_dotprod(v, r_hat, sync_dotprod, timers);
		DistrAsyncDotProds<Dtype> dp_s_s  = DistrAsyncDotProds<Dtype>::single_dotprod(s, s, sync_dotprod, timers);

		Grid3d<Dtype> * tt[] = {&t, &t};
		Grid3d<Dtype> * st[] = {&s, &t};
		DistrAsyncDotProds<Dtype> dp_ts_tt = DistrAsyncDotProds<Dtype>::multi_dotprod(tt, st, 2, sync_dotprod, timers);

		// Setup asynchronous stencil operations on vectors
		// NOTE: all these operators share the same CommHalo, so they can't be
		// overlapped. To overlap them safely, we'd need to create one CommHalo
		// object for each so that they don't share buffers
		DistrAsyncStencil<Dtype> stencil_x_r(A, x, r, comm_halo, sync_stencil, timers);
		DistrAsyncStencil<Dtype> stencil_ph_v(A, p_hat, v, comm_halo, sync_stencil, timers);
		DistrAsyncStencil<Dtype> stencil_sh_t(A, s_hat, t, comm_halo, sync_stencil, timers);

		// Setup slices
		Slice3d<Dtype> x_ = x.rank_cells;
		Slice3d<Dtype> b_ = b.rank_cells;
		Slice3d<Dtype> iM_ = iM.rank_cells;
		Slice3d<Dtype> r_ = r.rank_cells;
		Slice3d<Dtype> p_ = p.rank_cells;
		Slice3d<Dtype> s_ = s.rank_cells;
		Slice3d<Dtype> v_ = v.rank_cells;
		Slice3d<Dtype> t_ = t.rank_cells;
		Slice3d<Dtype> r_hat_ = r_hat.rank_cells;
		Slice3d<Dtype> p_hat_ = p_hat.rank_cells;
		Slice3d<Dtype> s_hat_ = s_hat.rank_cells;

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
			return BICGSTAB_SOLUTION_FOUND;
		}

		Dtype omega = 1.0;
		Dtype alpha = 0.0;
		Dtype rho_prev = 0.0;
		Dtype rho = 0.0;

		// Initialize r_hat
		slice_ops.copy(r_, r_hat_);
		
		int iter;
		Dtype s_norm2 = 0.0;
		for (iter=1; iter<maxiter+1; iter++) {
			// Compute rho = <r, r_hat>
			dp_r_rh.initiate();
			dp_r_rh.finalize(&rho);

			if (rho == 0.0) break;

			Dtype beta;
			if (iter > 1) {
				beta = (rho/rho_prev) * (alpha/omega);
				// p <- r + beta*(p - omega*v)
				slice_ops.combine3(1.0, r_, beta, p_, -beta*omega, v_, p_);
			} else {
				// p <- r
				slice_ops.copy(r_, p_);
			}

			// p_hat <- p ./ M
			slice_ops.multiply(p_, iM_, p_hat_);

			// Apply stencil: v <- A*p_hat
			stencil_ph_v.initiate();
			stencil_ph_v.compute_local();
			stencil_ph_v.finalize();

			// alpha <- rho / <v,r_hat>
			dp_v_rh.initiate();
			Dtype v_dot_rhat;
			dp_v_rh.finalize(&v_dot_rhat);
			alpha = rho / v_dot_rhat;

			// s <- r - alpha*v
			slice_ops.combine2(1.0, r_, -alpha, v_, s_);

			// Early convergence check on s
			dp_s_s.initiate();
			dp_s_s.finalize(&s_norm2);
			if (s_norm2 < tol*tol) {
				// x <- x + alpha*p_hat;
				slice_ops.combine2(1.0, x_, alpha, p_hat_, x_);
				break;
			}
    
			// s_hat <- s ./ M;
			slice_ops.multiply(s_, iM_, s_hat_);

			// Apply stencil: t <- A*s_hat
			stencil_sh_t.initiate();
			stencil_sh_t.compute_local();
			stencil_sh_t.finalize();

			// omega <- <t,s> / <t,t>
			dp_ts_tt.initiate();
			Dtype tdots_tdott[2];
			dp_ts_tt.finalize(tdots_tdott);
			omega = tdots_tdott[0] / tdots_tdott[1];

			// Update x
			// x <- x + alpha*p_hat + omega*s_hat
			slice_ops.combine3(1.0, x_, alpha, p_hat_, omega, s_hat_, x_);

			// r <- s - omega*t
			slice_ops.combine2(1.0, s_, -omega, t_, r_);

			// Compute square norm of r
			dp_r_r.initiate();
			dp_r_r.finalize(&r_norm2);

			// Check convergence
			err = sqrt(r_norm2/b_norm2);
			if (dd.myrank == 0) printf("BiCgStab: after iter=%d, err=%.10e\n", iter, err);
			if (err <= tol) break;
			if (omega == 0.0) break;

			rho_prev = rho;
		}

		// Now check the convergence status
		if (err <= tol) {
			*final_iter = iter;
			*final_err = err;
			return BICGSTAB_SOLUTION_FOUND;
		}

		if (s_norm2 <= tol*tol) {
			*final_iter = iter;
			*final_err = sqrt(s_norm2/b_norm2);
			return BICGSTAB_SOLUTION_FOUND;
		}

		if (omega == 0.0) {
			*final_iter = iter;
			*final_err = err;
			return BICGSTAB_BREAKDOWN_OMEGA;
		}

		if (rho == 0.0) {
			*final_iter = iter;
			*final_err = err;
			return BICGSTAB_BREAKDOWN_RHO;
		}

		assert(iter == maxiter+1);
		*final_iter = maxiter;
		*final_err = err;
		return BICGSTAB_REACHED_MAXITER;
	}
	
};

#endif
