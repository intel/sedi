#ifndef SLICE_3D_H
#define SLICE_3D_H

#ifdef WITH_HDF5
#include <hdf5.h>
#endif

template <typename Dtype>
class Slice3d {
	public:
	// Pointer to data buffer
	Dtype *data;

	// Number of elements along each dim, and total # of elements
	long n1, n2, n3, n;

	// Strides along dimensions 2 and 3 (stride in dimension 1 is 1)
	long s2, s3;

	Slice3d(Dtype *data_,
			long n1_, long n2_, long n3_,
			long s2_, long s3_) :
		data(data_),
		n1(n1_), n2(n2_), n3(n3_),
		n(n1*n2*n3),
		s2(s2_), s3(s3_) { }

	Slice3d() : data(00), n1(0), n2(0), n3(0), n(0), s2(0), s3(0) { }

	inline long element_idx(long i1, long i2, long i3) const {
		return i1 + i2*s2 + i3*s3;
	}

	inline Dtype * element_ptr(long i1, long i2, long i3) const {
		return data + element_idx(i1, i2, i3);
	}

	inline Dtype & element_ref(long i1, long i2, long i3) const {
		return data[element_idx(i1, i2, i3)];
	}

	void pack_into(Dtype * buf, long count) const {
		assert(count == n);
		long idx = 0;
		// Use row-major (C) ordering to easily interface with HDF5
		for (long i1=0; i1<n1; i1++) {
			for (long i2=0; i2<n2; i2++) {
				for (long i3=0; i3<n3; i3++) {
					buf[idx] = element_ref(i1, i2, i3);
					idx ++;
		} } }
	}

	void unpack_from(const Dtype * buf, long count) {
		assert(count == n);
		long idx = 0;
		// Use row-major (C) ordering to easily interface with HDF5
		for (long i1=0; i1<n1; i1++) {
			for (long i2=0; i2<n2; i2++) {
				for (long i3=0; i3<n3; i3++) {
					element_ref(i1, i2, i3) = buf[idx];
					idx ++;
		} } }
	}

	Slice3d<Dtype> extract_by_size(
			long ibeg1x, long ibeg2x, long ibeg3x,
			long n1x, long n2x, long n3x) const {
		return Slice3d<Dtype> (
				element_ptr(ibeg1x, ibeg2x, ibeg3x),
				n1x, n2x, n3x, s2, s3 );
	}

	Slice3d<Dtype> extract_by_index(
			long ibeg1x, long ibeg2x, long ibeg3x,
			long iend1x, long iend2x, long iend3x) const {
		return extract_by_size(
				ibeg1x, ibeg2x, ibeg3x,
				iend1x-ibeg1x, iend2x-ibeg2x, iend3x-ibeg3x );
	}

	void fill(Dtype value) {
		for (long i3=0; i3<n3; i3++) {
			for (long i2=0; i2<n2; i2++) {
				for (long i1=0; i1<n1; i1++) {
					const long idx = element_idx(i1, i2, i3);
					data[idx] = value;
				}
			}
		}
	}

	void accumulate(Dtype value) {
		for (long i3=0; i3<n3; i3++) {
			for (long i2=0; i2<n2; i2++) {
				for (long i1=0; i1<n1; i1++) {
					const long idx = element_idx(i1, i2, i3);
					data[idx] += value;
				}
			}
		}
	}

	static void copy(const Slice3d<Dtype> & src, Slice3d<Dtype> & dst) {
		assert( src.n1 == dst.n1 && src.n2 == dst.n2 && src.n3 == dst.n3 );
		for (long i3=0; i3<src.n3; i3++) {
			for (long i2=0; i2<src.n2; i2++) {
				for (long i1=0; i1<src.n1; i1++) {
					dst.element_ref(i1, i2, i3) = src.element_ref(i1, i2, i3);
		} } }
	}

	// Helper functions to dump/load a slice to/from HDF5

#ifdef WITH_HDF5
	// Using a filename (create new HDF5 file on demand)
	void dump_hdf5(const char * filename, const char * fieldname) {
		hid_t h_file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		dump_hdf5(h_file, fieldname);
		H5Fclose(h_file);
	}

	// Using an existing HDF5 file handle
	void dump_hdf5(hid_t h_file, const char * fieldname) {

		hsize_t dims[3];
		dims[0]=n1; dims[1]=n2; dims[2]=n3;

		hid_t h_dspace = H5Screate_simple(3, dims, 00);
		hid_t h_dset = H5Dcreate(h_file, fieldname, get_h5_datatype(),
				h_dspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

		// Pack data into a buffer with row-major order and write into dataset, see
		// https://www.hdfgroup.org/HDF5/doc/H5.intro.html#Intro-PMRdWrPortion
		Dtype * buf = new Dtype[n];
		pack_into(buf, n);
		H5Dwrite(h_dset, get_h5_datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
		delete [] buf;

		H5Dclose(h_dset);
		H5Sclose(h_dspace);
	}

	void load_hdf5(hid_t h_file, const char * fieldname) {
		hid_t h_dset = H5Dopen(h_file, fieldname, H5P_DEFAULT);
		hid_t h_dspace = H5Dget_space(h_dset);
		const int ndims = H5Sget_simple_extent_ndims(h_dspace);
		assert(ndims == 3);
		hsize_t dims[3];
		H5Sget_simple_extent_dims(h_dspace, dims, 00);
		assert(dims[0]==n1 && dims[1]==n2 && dims[2]==n3);

		Dtype * buf = new Dtype[n];
		H5Dread(h_dset, get_h5_datatype(), H5S_ALL, H5S_ALL, H5P_DEFAULT, buf);
		unpack_from(buf, n);
		delete [] buf;

		H5Sclose(h_dspace);
		H5Dclose(h_dset);
	}
#endif // WITH_HDF5

	private:
#ifdef WITH_HDF5
	hid_t get_h5_datatype();
#endif // WITH_HDF5
};

#ifdef WITH_HDF5
template <> hid_t Slice3d<float >::get_h5_datatype() { return H5T_NATIVE_FLOAT;  }
template <> hid_t Slice3d<double>::get_h5_datatype() { return H5T_NATIVE_DOUBLE; }
#endif // WITH_HDF5

#endif
