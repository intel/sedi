#ifndef COMM_HALO_H
#define COMM_HALO_H

#include <mpi.h>
#include <cassert>

#include "DomainDecomposition.hpp"
#include "Grid3d.hpp"
#include "Timers.hpp"

template <typename Dtype>
class CommHalo {
	public:
	typedef enum {
		COMM_SEND = 0,
		COMM_RECV = 1
	} comm_dir_t;
	
	// Memory region holding all comm buffers
	Dtype * commbuf;

	// Pointers to each ghost buffer
	// Indices are: (send/recv, direction, side)
	Dtype * ghost_bufs[2][3][2];
	long    ghost_size   [3]   ; // size of the ghost buffer in this direction

	// Array for MPI requests
	// Indices are: (send/recv, direction, side)
	MPI_Request reqs[2][3][2];

	// Array for MPI rank of neighbor
	// Indices are: (direction, side)
	int neighbor[3][2];

	DomainDecomposition & dd;
	long nghost;

	Timers * const timers;

	CommHalo(DomainDecomposition & dd_, long nghost_, Timers * timers_ = 00) :
		dd(dd_), nghost(nghost_), timers(timers_) {

		// Number of grid points in each ghost slab
		ghost_size[0] = nghost*dd.ngp2*dd.ngp3;
		ghost_size[1] = nghost*dd.ngp1*dd.ngp3;
		ghost_size[2] = nghost*dd.ngp1*dd.ngp2;

		const long buf_size = 2*(ghost_size[0] + ghost_size[1] + ghost_size[2])*2;
		commbuf = new Dtype[buf_size];

		// Set pointers for each ghost region and comm direction
		Dtype * tmpbuf = commbuf;
		for (int comm_dir=0; comm_dir<2; comm_dir++) { // send & recv direction
			ghost_bufs[comm_dir][0][0] = tmpbuf; tmpbuf += ghost_size[0];
			ghost_bufs[comm_dir][0][1] = tmpbuf; tmpbuf += ghost_size[0];

			ghost_bufs[comm_dir][1][0] = tmpbuf; tmpbuf += ghost_size[1];
			ghost_bufs[comm_dir][1][1] = tmpbuf; tmpbuf += ghost_size[1];

			ghost_bufs[comm_dir][2][0] = tmpbuf; tmpbuf += ghost_size[2];
			ghost_bufs[comm_dir][2][1] = tmpbuf; tmpbuf += ghost_size[2];
		}
		assert(tmpbuf-commbuf == buf_size);

		initialize_requests();

		// Set neighbor ranks
		// Note that we manage neighbor computations manually (i.e. without a
		// Cartesian MPI communicator) for GASPI compatibility
		// FIXME: could a Cartesian comm improve MPI performance?
		for (int dir=0; dir<3; dir++)
			for (int side=0; side<2; side++)
				neighbor[dir][side] = dd.get_neighbor_rank(dir, side);

	}

	void initialize_requests() {
		for (int comm_dir=0; comm_dir<2; comm_dir++)
			for (int dir=0; dir<3; dir++)
				for (int side=0; side<2; side++)
					reqs[comm_dir][dir][side] = MPI_REQUEST_NULL;
	}

	void post_all_receives() {
		if (timers) timers->start(Timers::TIMER_HALO_ISEND_IRECV);
		for (int dir=0; dir<3; dir++)
			for (int side=0; side<2; side++) {

				if (neighbor[dir][side] == MPI_PROC_NULL) continue;

				MPI_Irecv(
						ghost_bufs[COMM_RECV][dir][side],
						ghost_size[dir]*sizeof(Dtype), MPI_CHAR, // XXX: is this legal?
						neighbor[dir][side],
						42, MPI_COMM_WORLD,
						&(reqs[COMM_RECV][dir][side])
					);
			}
		if (timers) timers->stop(Timers::TIMER_HALO_ISEND_IRECV);
	}

	void pack_and_send_all(Grid3d<Dtype> & grid) {
		// Pack and send inner face cells (inside rank domain)
		for (int dir=0; dir<3; dir++)
			for (int side=0; side<2; side++) {

				if (neighbor[dir][side] == MPI_PROC_NULL) continue;

				if (timers) timers->start(Timers::TIMER_HALO_PACK);
				grid.inner_face_cells[dir][side].pack_into(
						ghost_bufs[COMM_SEND][dir][side], ghost_size[dir]);
				if (timers) timers->stop(Timers::TIMER_HALO_PACK);

				if (timers) timers->start(Timers::TIMER_HALO_ISEND_IRECV);
				MPI_Isend(
						ghost_bufs[COMM_SEND][dir][side],
						ghost_size[dir]*sizeof(Dtype), MPI_CHAR, // XXX: is this legal?
						neighbor[dir][side],
						42, MPI_COMM_WORLD,
						&(reqs[COMM_SEND][dir][side])
					);
				if (timers) timers->stop(Timers::TIMER_HALO_ISEND_IRECV);
			}
	}

	void wait_and_unpack_all(Grid3d<Dtype> & grid) {
		// Wait for incoming border cells from other ranks, and unpack them into ghosts

		int comm_done[2][3][2], unpack_done[3][2];

		// Initialize all completion flags
		for (int dir=0; dir<3; dir++) {
			for (int side=0; side<2; side++) {
				comm_done[COMM_SEND][dir][side] = 0;
				comm_done[COMM_RECV][dir][side] = 0;
				unpack_done[dir][side] = 0;
			}
		}

		int all_comms_done = 0;
		while (!all_comms_done) {

			all_comms_done = 1;

			if (timers) timers->start(Timers::TIMER_HALO_UNPACK_WAIT);
			for (int dir=0; dir<3; dir++)
				for (int side=0; side<2; side++) {

					if (neighbor[dir][side] == MPI_PROC_NULL) continue;

					// Test send
					if (! comm_done[COMM_SEND][dir][side])
						MPI_Test(
								&(reqs[COMM_SEND][dir][side]),
								&(comm_done[COMM_SEND][dir][side]),
								MPI_STATUS_IGNORE);

					// Test recv
					if (! comm_done[COMM_RECV][dir][side])
						MPI_Test(
								&(reqs[COMM_RECV][dir][side]),
								&(comm_done[COMM_RECV][dir][side]),
								MPI_STATUS_IGNORE);

					// If receive is ready and we need to unpack, do it
					if (comm_done[COMM_RECV][dir][side] && !unpack_done[dir][side]) {
						grid.ghost_cells[dir][side].unpack_from(
								ghost_bufs[COMM_RECV][dir][side], ghost_size[dir]);
						unpack_done[dir][side] = 1;
					}

					all_comms_done = all_comms_done &&
								comm_done[COMM_SEND][dir][side] &&
								comm_done[COMM_RECV][dir][side];
				}
			if (timers) timers->stop(Timers::TIMER_HALO_UNPACK_WAIT);
		}

		// Reset requests
		initialize_requests();
	}

	~CommHalo() {
		delete [] commbuf;
		commbuf = 0;
	}
	
};

#endif
