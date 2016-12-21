#ifndef DOMAIN_DECOMPOSITION_H
#define DOMAIN_DECOMPOSITION_H

#include <cassert>
#include <cstdio>

class DomainDecomposition {
	public:
	
	// Dimensions of global grid and number of grid points
	long ng1, ng2, ng3, ng;

	// Shape of process decomposition
	int np1, np2, np3, np;

	// Dimensions of process-local grid
	long ngp1, ngp2, ngp3, ngp;

	// My process id
	int myrank;

	DomainDecomposition(
			long ng1_, long ng2_, long ng3_,
			int np1_, int np2_, int np3_) :
		ng1(ng1_), ng2(ng2_), ng3(ng3_), ng(ng1_*ng2_*ng3_),
		np1(np1_), np2(np2_), np3(np3_), np(np1_*np2_*np3_)
	{
		int nranks;
		MPI_Comm_size(MPI_COMM_WORLD, &nranks);
		assert(nranks == np);

		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

		assert(ng1 % np1 == 0);
		assert(ng2 % np2 == 0);
		assert(ng3 % np3 == 0);

		ngp1 = ng1 / np1;
		ngp2 = ng2 / np2;
		ngp3 = ng3 / np3;
		ngp = ngp1*ngp2*ngp3;

		if (myrank == 0) {
			printf("Global grid shape is (%ld, %ld, %ld)\n", ng1, ng2, ng3);
			printf("Domain decomposition across %d MPI ranks\n", np);
			printf("Processor decomposition is (%d, %d, %d)\n", np1, np2, np3);
			printf("Process block shape is (%ld, %ld, %ld)\n", ngp1, ngp2, ngp3);
		}
	}

	void rank_1d_to_3d(int rank, int *i1, int *i2, int *i3) const {
		*i1 = rank % np1;
		*i2 = (rank/np1) % np2;
		*i3 = rank / (np1*np2);
	}

	void rank_3d_to_1d(int i1, int i2, int i3, int *rank) const {
		*rank = i1 + np1*i2 + np1*np2*i3;
	}

	int get_neighbor_rank(int dir, int side) const {
		// My rank -> 3D index
		int idx3d[3];
		rank_1d_to_3d(myrank, &(idx3d[0]),  &(idx3d[1]),  &(idx3d[2]));

		// Which direction do we shift?
		int shift = (side == 0 ? -1 : 1);
		assert(dir >= 0 && dir < 3);
		idx3d[dir] += shift;

		// Make sure there's a processor there
		int np[3] = {np1, np2, np3};
		if (idx3d[dir] < 0 || idx3d[dir] >= np[dir]) // No neighbor there
			return MPI_PROC_NULL;

		// Convert shifted 3D index back to rank id
		int nb_rank;
		rank_3d_to_1d(idx3d[0], idx3d[1], idx3d[2], &nb_rank);
		return nb_rank;
	}

	void index_global_to_local(long iglob1, long iglob2, long iglob3,
			long *iloc1, long *iloc2, long *iloc3, int *rank) {
		assert(0 <= iglob1 && iglob1 < ng1);
		assert(0 <= iglob2 && iglob2 < ng2);
		assert(0 <= iglob3 && iglob3 < ng3);

		int p1 = iglob1 / ngp1;
		int p2 = iglob2 / ngp2;
		int p3 = iglob3 / ngp3;
		rank_3d_to_1d(p1, p2, p3, rank);

		*iloc1 = (iglob1 % ngp1);
		*iloc2 = (iglob2 % ngp2);
		*iloc3 = (iglob3 % ngp3);
	}

	void index_local_to_global(long iloc1, long iloc2, long iloc3, int rank,
			long *iglob1, long *iglob2, long *iglob3) {
		assert(0 <= iloc1 && iloc1 < ngp1);
		assert(0 <= iloc2 && iloc2 < ngp2);
		assert(0 <= iloc3 && iloc3 < ngp3);

		int p1, p2, p3;
		rank_1d_to_3d(rank, &p1, &p2, &p3);
		*iglob1 = iloc1 + p1*ngp1;
		*iglob2 = iloc2 + p2*ngp2;
		*iglob3 = iloc3 + p3*ngp3;
	}

};

#endif
