#ifndef GRID_3D_H
#define GRID_3D_H

#include "DomainDecomposition.hpp"
#include "Slice3d.hpp"

template <typename Dtype>
class Grid3d {
	public:

	// Pointer to vector data
	Dtype* data;

	// Size of ghost layer
	long nghost;

	// Size of inner layer (typically = nghost if ghost cells are allocated)
	long ninner;

	// Logical and allocated dimensions and size
	long n1, n2, n3, n;
	long nalloc1, nalloc2, nalloc3, nalloc;

	// Strides along dimension 2 and 3
	long s2, s3;

	Slice3d<Dtype> all_cells, rank_cells, inner_cells;
	Slice3d<Dtype> inner_face_cells[3][2], border_cells[3][2], ghost_cells[3][2];

	Grid3d(DomainDecomposition & dd, long nghost_, long ninner_) :
		nghost(nghost_),
		ninner(ninner_),
		n1(dd.ngp1),
		n2(dd.ngp2),
		n3(dd.ngp3),
		n(n1*n2*n3),
		nalloc1(n1+2*nghost),
		nalloc2(n2+2*nghost),
		nalloc3(n3+2*nghost),
		nalloc(nalloc1*nalloc2*nalloc3),
		s2(nalloc1),
		s3(nalloc1*nalloc2)
	{
		// Allocate vector memory
		data = new Dtype[nalloc];

		// Slice on all the cells
		all_cells = Slice3d<Dtype> (data, nalloc1, nalloc2, nalloc3, s2, s3);

		// Slice on the "active" cells (belonging to my rank)
		rank_cells = all_cells.extract_by_size(nghost, nghost, nghost, n1, n2, n3);

		// Inner cells (active cells not at face, edge or corner)
		inner_cells = rank_cells.extract_by_size(ninner, ninner, ninner, n1-2*ninner, n2-2*ninner, n3-2*ninner);

		// Ghost cells slices
		ghost_cells[0][0] = rank_cells.extract_by_size(-nghost, 0, 0,    nghost, n2, n3);
		ghost_cells[0][1] = rank_cells.extract_by_size(     n1, 0, 0,    nghost, n2, n3);

		ghost_cells[1][0] = rank_cells.extract_by_size(0, -nghost, 0,    n1, nghost, n3);
		ghost_cells[1][1] = rank_cells.extract_by_size(0,      n2, 0,    n1, nghost, n3);

		ghost_cells[2][0] = rank_cells.extract_by_size(0, 0, -nghost,    n1, n2, nghost);
		ghost_cells[2][1] = rank_cells.extract_by_size(0, 0,      n3,    n1, n2, nghost);

		// Inner face cells: active cells on a face (but not edge or corner),
		// maps directly to ghost cells for distributed memory communications
		inner_face_cells[0][0] = rank_cells.extract_by_size(0        , 0, 0,    ninner, n2, n3);
		inner_face_cells[0][1] = rank_cells.extract_by_size(n1-ninner, 0, 0,    ninner, n2, n3);

		inner_face_cells[1][0] = rank_cells.extract_by_size(0,         0, 0,    n1, ninner, n3);
		inner_face_cells[1][1] = rank_cells.extract_by_size(0, n2-ninner, 0,    n1, ninner, n3);

		inner_face_cells[2][0] = rank_cells.extract_by_size(0, 0,         0,    n1, n2, ninner);
		inner_face_cells[2][1] = rank_cells.extract_by_size(0, 0, n3-ninner,    n1, n2, ninner);

		// Border cells: active cells on a face, edge or corner
		// Borders along dimensions 1 and 2 are staggered
		border_cells[0][0] = rank_cells.extract_by_size(0        , ninner, ninner,    ninner, n2-ninner, n3-2*ninner);
		border_cells[0][1] = rank_cells.extract_by_size(n1-ninner, 0     , ninner,    ninner, n2-ninner, n3-2*ninner);

		border_cells[1][0] = rank_cells.extract_by_size(0     , 0        , ninner,    n1-ninner, ninner, n3-2*ninner);
		border_cells[1][1] = rank_cells.extract_by_size(ninner, n2-ninner, ninner,    n1-ninner, ninner, n3-2*ninner);

		// Borders along dimension 3 form "caps" of the cube, and have
		// potentially more cells than the other border slices.
		// However, these slices are contiguous in memory, and performance
		// should therefore be better when operating on these slices, helping
		// reduce the imbalance.
		border_cells[2][0] = rank_cells.extract_by_size(0, 0, 0,            n1, n2, ninner);
		border_cells[2][1] = rank_cells.extract_by_size(0, 0, n3-ninner,    n1, n2, ninner);

	}

	void fill_ghosts(Dtype value) {
		for (int dir=0; dir<3; dir++)
			for (int side=0; side<2; side++)
				ghost_cells[dir][side].fill(value);
	}

	~Grid3d() {
		delete [] data;
		data = 0;
	}

};

#endif
