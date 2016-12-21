#ifndef DISTR_ASYNC_DOT_PRODS_H
#define DISTR_ASYNC_DOT_PRODS_H

#include <cassert>
#include <cstdio>

#include <mpi.h>

#include "Slice3d.hpp"
#include "Grid3d.hpp"
#include "SliceOps.hpp"
#include "Timers.hpp"
#include "BlockSliceIterator.hpp"
#include "Platform.hpp"

// A class to compute the dot products of two series of Grid3d objects xs and ys
// asynchronously in distributed memory
template <typename Dtype>
class DistrAsyncDotProds {
	public:

	static const int ndotprod_max = 8;

	// Vector of source grids
	int ndotprod;
	Grid3d<Dtype> *xs[ndotprod_max];
	Grid3d<Dtype> *ys[ndotprod_max];

	Dtype result[ndotprod_max], result_loc[ndotprod_max]; // Global and local value of dot products

#ifdef WITH_MPI3
	MPI_Request req;
#endif

	bool synchronous;

	Timers * const timers;
	SliceOps<Dtype> slice_ops;

	// Track stage of async comm to make sure we don't do anything stupid
	typedef enum {
		STAGE_READY = 1,
		STAGE_INITIATED = 2
	} stage_t;
	stage_t stage;

	// Block iterator for cache blocking
	BlockSliceIterator<Dtype> * block_iter;

	DistrAsyncDotProds(Grid3d<Dtype> ** xs_, Grid3d<Dtype> ** ys_, int ndotprod_, bool synchronous_, Timers * timers_ = 00):
		ndotprod(ndotprod_),
#ifdef WITH_MPI3
		req(MPI_REQUEST_NULL),
#endif
		synchronous(synchronous_),
		timers(timers_),
		slice_ops(timers),
		stage(STAGE_READY) {

		assert(ndotprod <= ndotprod_max);
		assert(ndotprod > 0);

		for (int i=0; i<ndotprod; i++) {
			xs[i] = xs_[i];
			ys[i] = ys_[i];
		}

		for (int i=0; i<ndotprod; i++) {
			result    [i] = 0.0;
			result_loc[i] = 0.0;
		}

		block_iter = new BlockSliceIterator<Dtype>(xs[0]->rank_cells, -1, -1, -1);
		if (ndotprod >= 2) {
			// Use cache blocking across arrays
			*block_iter = BlockSliceIterator<Dtype>::cache_aware_iterator(
					xs[0]->rank_cells, PLATFORM_L3_BYTES_PER_CORE, 2*ndotprod);
		}
	}

	~DistrAsyncDotProds() {
		delete block_iter;
	}

	void initiate() {
		assert(stage == STAGE_READY);
		// Compute local dot products of the active cells for each grid pair in xs, ys

		for (int i=0; i<ndotprod; i++)
			result_loc[i] = 0.0;

		for(block_iter->reset(); !block_iter->end(); block_iter->next()) {
			for (int i=0; i<ndotprod; i++) {
				Slice3d<Dtype> xblock = block_iter->get_block(xs[i]->rank_cells);
				Slice3d<Dtype> yblock = block_iter->get_block(ys[i]->rank_cells);
				result_loc[i] += slice_ops.dotprod(xblock, yblock);
			}
		}

		if (timers) timers->start(Timers::TIMER_COMM_DOTPROD);
#ifdef WITH_MPI3
		// Initiate nonblocking collective using request req, if we're not in synchronous mode
		if (!synchronous)
			MPI_Iallreduce(result_loc, result, ndotprod, get_mpi_datatype(), MPI_SUM, MPI_COMM_WORLD, &req);
		else
			MPI_Allreduce(result_loc, result, ndotprod, get_mpi_datatype(), MPI_SUM, MPI_COMM_WORLD);
#else
		// Just do a blocking collective here
		MPI_Allreduce(result_loc, result, ndotprod, get_mpi_datatype(), MPI_SUM, MPI_COMM_WORLD);
#endif
		if (timers) timers->stop(Timers::TIMER_COMM_DOTPROD);

		stage = STAGE_INITIATED;
	}

	void finalize(Dtype * ret) {
		assert(stage == STAGE_INITIATED);

#ifdef WITH_MPI3
		// Now wait for end of nonblocking collective, if not in synchronous mode
		if (!synchronous) {
			if (timers) timers->start(Timers::TIMER_COMM_DOTPROD);
			MPI_Wait(&req, MPI_STATUS_IGNORE);
			if (timers) timers->stop(Timers::TIMER_COMM_DOTPROD);
			req = MPI_REQUEST_NULL;
		}
#else
		// We did a blocking collective in initiate() so nothing to do here...
#endif

		stage = STAGE_READY;

		// Copy back result into output array
		for (int i=0; i<ndotprod; i++)
			ret[i] = result[i];
	}

	static DistrAsyncDotProds<Dtype> single_dotprod(Grid3d<Dtype> & x, Grid3d<Dtype> & y, bool synchronous, Timers * timers = 00) {
		Grid3d<Dtype> * xaddr = &x;
		Grid3d<Dtype> * yaddr = &y;
		DistrAsyncDotProds<Dtype> dp(&xaddr, &yaddr, 1, synchronous, timers);
		return dp;
	}

	static DistrAsyncDotProds<Dtype> multi_dotprod(Grid3d<Dtype> ** xs, Grid3d<Dtype> ** ys, int n, bool synchronous, Timers * timers = 00) {
		DistrAsyncDotProds<Dtype> dp(xs, ys, n, synchronous, timers);
		return dp;
	}

	private:
	MPI_Datatype get_mpi_datatype();
};

template <> MPI_Datatype DistrAsyncDotProds<float >::get_mpi_datatype() { return MPI_FLOAT;  }
template <> MPI_Datatype DistrAsyncDotProds<double>::get_mpi_datatype() { return MPI_DOUBLE; }

#endif
