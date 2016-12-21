#ifndef DISTR_ASYNC_STENCIL_H
#define DISTR_ASYNC_STENCIL_H

#include <cassert>

#include "Stencil.hpp"
#include "CommHalo.hpp"
#include "Grid3d.hpp"
#include "Timers.hpp"

// A class to compute y <- stencil(x) with comm/compute overlap
// in distributed memory
template <typename Dtype>
class DistrAsyncStencil {
	public:
	Stencil<Dtype> & stencil;
	CommHalo<Dtype> & comm_halo;

	// Source and destination grids
	Grid3d<Dtype> & x;
	Grid3d<Dtype> & y;

	// Whether to use synchronous comms
	const bool synchronous;

	Timers * const timers;

	// Track stage of async comm to make sure we don't do anything stupid
	typedef enum {
		STAGE_READY = 1,
		STAGE_INITIATED = 2,
		STAGE_COMPUTED_LOC = 3
	} stage_t;
	stage_t stage;

	DistrAsyncStencil(
			Stencil<Dtype> & stencil_,
			Grid3d<Dtype> & x_, Grid3d<Dtype> & y_,
			CommHalo<Dtype> & comm_halo_,
			bool synchronous_,
			Timers * timers_ = 00) :
		stencil(stencil_), comm_halo(comm_halo_), x(x_), y(y_),
		synchronous(synchronous_),
		timers(timers_),
		stage(STAGE_READY) {
		// Check that we didn't pass a same grid as x and y, as this will cause trouble
		assert( &x != &y );
	}

	~DistrAsyncStencil() { }

	void initiate() {
		assert(stage == STAGE_READY);
		if (!synchronous) {
			// Post receives for ghost regions
			comm_halo.post_all_receives();
			// Post sends for border regions to fill in ghosts on other ranks
			comm_halo.pack_and_send_all(x);
		} // If synchronous, do nothing!
		stage = STAGE_INITIATED;
	}

	void compute_local() {
		assert(stage == STAGE_INITIATED);
		if (!synchronous) {
			// Compute locally on inner cells, for which we don't need the ghosts
			if (timers) timers->start(Timers::TIMER_COMPUTE_STENCIL_INNER);
			stencil.apply(x.inner_cells, y.inner_cells);
			if (timers) timers->stop(Timers::TIMER_COMPUTE_STENCIL_INNER);
		} // If synchronous, do nothing!
		stage = STAGE_COMPUTED_LOC;
	}

	void finalize() {
		assert(stage == STAGE_COMPUTED_LOC);

		if (!synchronous) {
			// Wait and unpack ghost data
			comm_halo.wait_and_unpack_all(x);

			// Now compute stencil at inner border face cells
			if (timers) timers->start(Timers::TIMER_COMPUTE_STENCIL_BORDER);
			for (int dir=0; dir<3; dir++)
				for (int side=0; side<2; side++)
					stencil.apply(x.border_cells[dir][side], y.border_cells[dir][side]);
			if (timers) timers->stop(Timers::TIMER_COMPUTE_STENCIL_BORDER);
		} else {
			// Synchronous mode: now we need to do everything :)
			// Communicate to update the halos
			comm_halo.post_all_receives();
			comm_halo.pack_and_send_all(x);
			comm_halo.wait_and_unpack_all(x);
			// Now apply the stencil on the whole grid
			if (timers) timers->start(Timers::TIMER_COMPUTE_STENCIL_INNER);
			stencil.apply(x.rank_cells, y.rank_cells);
			if (timers) timers->stop(Timers::TIMER_COMPUTE_STENCIL_INNER);
		}

		stage = STAGE_READY;
	}

};

#endif
