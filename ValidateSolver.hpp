#ifndef VALIDATE_SOLVER_H
#define VALIDATE_SOLVER_H

#ifndef WITH_HDF5
#error "Cannot compile ValidateSolver without HDF5, and WITH_HDF5 is not set!"
#endif

#include <cassert>
#include <hdf5.h>
#include <mpi.h>

#include "DomainDecomposition.hpp"
#include "Grid3d.hpp"


template <typename Dtype>
class ValidateSolver {
	public:

	DomainDecomposition & dd;
	const char * ref_h5file; // HDF5 file for reference input and output

	Dtype * buf_sol; // Buffer for solution
	Dtype * buf_rhs; // Buffer for RHS
	Slice3d<Dtype> glob_sol, glob_rhs; // solution and rhs assembled for all ranks
	Slice3d<Dtype> my_sol  , my_rhs;   // solution and rhs slice just for current rank

	double l2_sol, linf_sol; // Norms of reference solution

	ValidateSolver(DomainDecomposition & dd_, const char * ref_h5file_) :
		dd(dd_), ref_h5file(ref_h5file_) {

		hid_t h_file = H5Fopen(ref_h5file, H5F_ACC_RDONLY, H5P_DEFAULT);

		// Retrieve dims of validation data set
		hid_t h_dset = H5Dopen(h_file, "x_k", H5P_DEFAULT);
		hid_t h_dspace = H5Dget_space(h_dset);
		const int ndims = H5Sget_simple_extent_ndims(h_dspace);
		assert(ndims == 3);
		hsize_t dims[3];
		H5Sget_simple_extent_dims(h_dspace, dims, 00);
		H5Sclose(h_dspace);
		H5Dclose(h_dset);

		// Check against the current grid size we're using
		assert(dd.ng1==dims[0] && dd.ng2==dims[1] && dd.ng3==dims[2]);

		// Total buffer size to allocate
		const long n = ((long)dims[0])*((long)dims[1])*((long)dims[2]);

		// Allocate space and setup slices
		buf_sol = new Dtype[n];
		buf_rhs = new Dtype[n];
		glob_sol = Slice3d<Dtype>(buf_sol,   dd.ng1, dd.ng2, dd.ng3,   dd.ng1, dd.ng1*dd.ng2);
		glob_rhs = Slice3d<Dtype>(buf_rhs,   dd.ng1, dd.ng2, dd.ng3,   dd.ng1, dd.ng1*dd.ng2);

		// Read HDF5 arrays into newly created slices
		glob_sol.load_hdf5(h_file, "x_k");
		glob_rhs.load_hdf5(h_file, "v_k");

		// Compute norms of reference solution
		l2_sol = linf_sol = 0.0;
		for (long i3=0; i3<glob_sol.n3; i3++) {
			for (long i2=0; i2<glob_sol.n2; i2++) {
				for (long i1=0; i1<glob_sol.n1; i1++) {
					double x = glob_sol.element_ref(i1, i2, i3);
					l2_sol += x*x;
					double abs_x = fabs(x);
					if (abs_x > linf_sol) linf_sol = abs_x;
		} } }
		l2_sol = sqrt( l2_sol/glob_sol.n );

		// Extract slices for myrank (just our slab in the global grid from the domain decomp)
		long ig1, ig2, ig3;
		dd.index_local_to_global(0, 0, 0, dd.myrank, &ig1, &ig2, &ig3);
		my_sol = glob_sol.extract_by_size(ig1, ig2, ig3, dd.ngp1, dd.ngp2, dd.ngp3);
		my_rhs = glob_rhs.extract_by_size(ig1, ig2, ig3, dd.ngp1, dd.ngp2, dd.ngp3);

		printf("VALIDATION: rank %d read arrays from %s\n", dd.myrank, ref_h5file);

		H5Fclose(h_file);

	};

	~ValidateSolver() {
		delete [] buf_sol;
		delete [] buf_rhs;
	}

	void fill_with_ref_rhs(Slice3d<Dtype> & x) {
		if (dd.myrank == 0) printf("VALIDATION: filling RHS from validation dataset\n");
		Slice3d<Dtype>::copy(my_rhs, x);
	}

	void compare_to_ref_solution(Slice3d<Dtype> & x) {
		if (dd.myrank == 0) printf("VALIDATION: comparing to ref solution from validation dataset\n");
		double err_2 = 0.0, err_inf = 0.0;
		for (long i3=0; i3<x.n3; i3++) {
			for (long i2=0; i2<x.n2; i2++) {
				for (long i1=0; i1<x.n1; i1++) {
					double delta = x.element_ref(i1, i2, i3) - my_sol.element_ref(i1, i2, i3);
					err_2 += delta*delta;
					double abs_delta = fabs(delta);
					if (abs_delta > err_inf) err_inf = abs_delta;
		} } }

		double err_2_glob = 0.0, err_inf_glob = 0.0;
		MPI_Allreduce(&err_inf, &err_inf_glob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(&err_2  , &err_2_glob  , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		err_2_glob = sqrt( err_2_glob/dd.ng );
		if (dd.myrank == 0) printf("VALIDATION: rel L2 error=%.4e, rel Linf error=%.4e\n",
				err_2_glob/l2_sol, err_inf_glob/linf_sol);
	}
	
};

#endif
