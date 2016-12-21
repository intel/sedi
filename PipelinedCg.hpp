#ifndef PIPELINED_CG_H
#define PIPELINED_CG_H

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
#include "BlockSliceIterator.hpp"
#include "Slice3d.hpp"

#include "Timers.hpp"

template <typename Dtype>
class PipelinedCg {
	public:
	DomainDecomposition & dd;

	// Temp vectors for pipelined CG
	Grid3d<Dtype> r, w, n, z, q, s, p;

	// These vectors need ghost because the stencil gets applied to them
	Grid3d<Dtype> u, m;

	typedef enum {
		PLCG_SOLUTION_FOUND = 0,
		PLCG_REACHED_MAXITER = 1
	} result_t;

	typedef enum {
		PLCG_ASYNCHRONOUS = 0,
		PLCG_SYNCHRONOUS_STENCIL = 1 << 0,
		PLCG_SYNCHRONOUS_DOTPROD = 1 << 1
	} mode_t;

	// Whether to do the dotprods and stencils in a synchronous way
	const bool sync_dotprod, sync_stencil;

	Timers * const timers;
	SliceOps<Dtype> slice_ops;

	PipelinedCg(DomainDecomposition & dd_, long nghost, int mode, Timers *timers_ = 00) :
		dd(dd_),
		r(dd, 0, nghost), w(dd, 0, nghost), n(dd, 0, nghost), 
		z(dd, 0, nghost), q(dd, 0, nghost), s(dd, 0, nghost), p(dd, 0, nghost),
		// Vectors with ghosts
		u(dd, nghost, nghost), m(dd, nghost, nghost),
		sync_dotprod((mode & PLCG_SYNCHRONOUS_DOTPROD) != 0),
		sync_stencil((mode & PLCG_SYNCHRONOUS_STENCIL) != 0),
		timers(timers_), slice_ops(timers)
	{
		// Make sure ghosts are set to 0, otherwise we will be picking up
		// affine terms in the stencil. Some of these ghosts will be updated by
		// MPI comms, but others correspond to global domain boundaries and
		// will only be read from now on (which is why they have to be set to 0).
		// XXX: this may need refactoring/adapting to other operators
		u.fill_ghosts(0.0);
		m.fill_ghosts(0.0);

		if (dd.myrank == 0) {
			printf("Initialized PipelinedCg solver, sync_dotprod=%d, sync_stencil=%d\n",
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
		DistrAsyncDotProds<Dtype> dp_r_r = DistrAsyncDotProds<Dtype>::single_dotprod(r, r, sync_dotprod, timers);

		Grid3d<Dtype> * rwr[] = {&r, &w, &r};
		Grid3d<Dtype> * uur[] = {&u, &u, &r};
		DistrAsyncDotProds<Dtype> dp_ru_wu_rr = DistrAsyncDotProds<Dtype>::multi_dotprod(rwr, uur, 3, sync_dotprod, timers);

		// Setup asynchronous stencil operations on vectors
		// NOTE: all these operators share the same CommHalo, so they can't be
		// overlapped. To overlap them safely, we'd need to create one CommHalo
		// object for each so that they don't share buffers
		DistrAsyncStencil<Dtype> stencil_x_r(A, x, r, comm_halo, sync_stencil, timers);
		DistrAsyncStencil<Dtype> stencil_u_w(A, u, w, comm_halo, sync_stencil, timers);
		DistrAsyncStencil<Dtype> stencil_m_n(A, m, n, comm_halo, sync_stencil, timers);

		// Block iterator
		BlockSliceIterator<Dtype> block_iter = BlockSliceIterator<Dtype>::cache_aware_iterator(x.rank_cells, PLATFORM_L3_BYTES_PER_CORE, 20);

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
		slice_ops.combine2(1.0, b.rank_cells, -1.0, r.rank_cells, r.rank_cells);
		
		// Compute <r,r>
		dp_r_r.initiate();
		Dtype r_norm2;
		dp_r_r.finalize(&r_norm2);
		Dtype err = sqrt(r_norm2/b_norm2);
		if (err < tol) {
			*final_iter = 0;
			*final_err = err;
			return PLCG_SOLUTION_FOUND;
		}

		Dtype alpha = 0.0;
		Dtype beta = 0.0;
		Dtype gamma = 0.0;
		Dtype gamma_old = 0.0;
		Dtype delta = 0.0;

		// Initialize u
		// u <- r ./ M
		slice_ops.multiply(r.rank_cells, iM.rank_cells, u.rank_cells);

		// w <- A*u
		stencil_u_w.initiate();
		stencil_u_w.compute_local();
		stencil_u_w.finalize();

		// Initialize z, q, s, p
		z.all_cells.fill(0.0);
		q.all_cells.fill(0.0);
		s.all_cells.fill(0.0);
		p.all_cells.fill(0.0);
		
		int iter;
		for (iter=1; iter<maxiter+1; iter++) {

			// Initiate dot products for gamma, delta and residual
			dp_ru_wu_rr.initiate();

			// m <- w ./ M
			slice_ops.multiply(w.rank_cells, iM.rank_cells, m.rank_cells);

			// Initiate stencil for n <- A*m
			stencil_m_n.initiate();

			// Compute stencil locally
			stencil_m_n.compute_local();

			// We need gamma and delta: finalize dot prods
			gamma_old = gamma;

			Dtype gamma_delta_rnorm2[3];
			dp_ru_wu_rr.finalize(gamma_delta_rnorm2);
			gamma   = gamma_delta_rnorm2[0];
			delta   = gamma_delta_rnorm2[1];
			r_norm2 = gamma_delta_rnorm2[2];

			if (iter > 1) {
				beta = gamma/gamma_old;
				alpha = 1.0 / (delta/gamma - beta/alpha);
			} else {
				beta = 0.0;
				alpha = gamma/delta;
			}

			// We need n: finalize stencil
			stencil_m_n.finalize();

			for(block_iter.reset(); !block_iter.end(); block_iter.next()) {

			Slice3d<Dtype> x_ = block_iter.get_block(x.rank_cells);
			Slice3d<Dtype> r_ = block_iter.get_block(r.rank_cells);
			Slice3d<Dtype> w_ = block_iter.get_block(w.rank_cells);
			Slice3d<Dtype> n_ = block_iter.get_block(n.rank_cells);
			Slice3d<Dtype> z_ = block_iter.get_block(z.rank_cells);
			Slice3d<Dtype> q_ = block_iter.get_block(q.rank_cells);
			Slice3d<Dtype> s_ = block_iter.get_block(s.rank_cells);
			Slice3d<Dtype> p_ = block_iter.get_block(p.rank_cells);
			Slice3d<Dtype> u_ = block_iter.get_block(u.rank_cells);
			Slice3d<Dtype> m_ = block_iter.get_block(m.rank_cells);

			// q <- m + beta*q
			slice_ops.combine2(1.0, m_, beta, q_, q_);

			// s <- w + beta*s
			slice_ops.combine2(1.0, w_, beta, s_, s_);

			// p <- u + beta*p
			slice_ops.combine2(1.0, u_, beta, p_, p_);

			// z <- n + beta*z
			slice_ops.combine2(1.0, n_, beta, z_, z_);

			// r <- r - alpha*s
			slice_ops.combine2(1.0, r_, -alpha, s_, r_);

			// x <- x + alpha*p
			slice_ops.combine2(1.0, x_, alpha, p_, x_);

			// u <- u - alpha*q
			slice_ops.combine2(1.0, u_, -alpha, q_, u_);

			// w <- w - alpha*z
			slice_ops.combine2(1.0, w_, -alpha, z_, w_);

			}


			// Check convergence
			err = sqrt(r_norm2/b_norm2);
			if (dd.myrank == 0) printf("PipelinedCg: after iter=%d, err=%.10e\n", iter, err);
			if (err <= tol) break;
		}

		// Now check the convergence status
		if (err <= tol) {
			*final_iter = iter;
			*final_err = err;
			return PLCG_SOLUTION_FOUND;
		}

		assert(iter == maxiter+1);
		*final_iter = maxiter;
		*final_err = err;
		return PLCG_REACHED_MAXITER;
	}
	
};

#endif
