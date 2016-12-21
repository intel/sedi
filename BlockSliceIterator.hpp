#ifndef BLOCK_SLICE_ITERATOR_H
#define BLOCK_SLICE_ITERATOR_H

#include "Slice3d.hpp"
#include <cassert>

template <typename Dtype>
class BlockSliceIterator {
	public:

	// Base slice to use as iteration space
	// Only dimensions are used, not the data pointer
	Slice3d<Dtype> base_slice;

	// Block size in each direction
	long bs1, bs2, bs3;

	// Number of blocks along each direction
	long nb1, nb2, nb3;
	long nb;

	// Current iterator block index
	long ib1, ib2, ib3;
	long ib;

	// Current iterator block element begin/end indices
	long ibeg1, ibeg2, ibeg3;
	long iend1, iend2, iend3;

	BlockSliceIterator(Slice3d<Dtype> base_slice_, long bs1_, long bs2_, long bs3_) :
		base_slice(base_slice_)
	{
		// Ensure sane values of block sizes
		bs1 = block_size(base_slice.n1, bs1_);
		bs2 = block_size(base_slice.n2, bs2_);
		bs3 = block_size(base_slice.n3, bs3_);

		// Compute number of blocks along each direction
		nb1 = num_blocks(base_slice.n1, bs1);
		nb2 = num_blocks(base_slice.n2, bs2);
		nb3 = num_blocks(base_slice.n3, bs3);
		nb = nb1*nb2*nb3;

		// Initialize iterator index
		reset();
	}

	Slice3d<Dtype> get_block(const Slice3d<Dtype> & slice) {
		assert(slice.n1 == base_slice.n1 && slice.n2 == base_slice.n2 && slice.n3 == base_slice.n3);
		return slice.extract_by_index(ibeg1, ibeg2, ibeg3, iend1, iend2, iend3);
	}

	void reset() {
		ib1 = ib2 = ib3 = 0;
		ib = 0;
		// Compute ibeg*, iend* for current block index
		update_ibeg_iend();
	}

	void next() {
		// Move to next block
		ib1++;
		if (ib1 == nb1) {
			ib1 = 0;
			ib2++;
			if (ib2 == nb2) {
				ib2 = 0;
				ib3++;
			}
		}
		ib++;
		// Compute ibeg*, iend* for current block index
		update_ibeg_iend();
	}

	bool end() {
		return (ib == nb);
	}

	// Helper for cache-aware blocking iterator, with n arrays in cache
	static BlockSliceIterator<Dtype> cache_aware_iterator(Slice3d<Dtype> base_slice, long cachesize, int narrays) {
		const float safety_factor = 0.6;
		long eff_cachesize = (cachesize / narrays) * safety_factor;
		long nelem = eff_cachesize / sizeof(Dtype);
		assert(nelem > 0);

		// Try not to cut in direction 1, for PF, vectorization, ...
		long bs1_try = base_slice.n1;
		for (long nb1_try=1; nb1_try<=base_slice.n1; nb1_try++) {
			bs1_try = base_slice.n1/nb1_try;
			if (bs1_try < nelem) break;
		}

		// Try a small block size along direction 2
		// Start at 10, and go down to 1
		long bs2_try = 10;
		for (; bs2_try>=1; bs2_try--) {
			if (bs1_try*bs2_try < nelem) break;
		}

		// Pick largest possible block size in direction 3 that fits
		long bs3_try = base_slice.n3;
		for (long nb3_try=1; nb3_try<=base_slice.n3; nb3_try++) {
			bs3_try = base_slice.n3/nb3_try;
			if (bs1_try*bs2_try*bs3_try < nelem) break;
		}

		return BlockSliceIterator<Dtype>(base_slice, bs1_try, bs2_try, bs3_try);
	}

	private:

	long block_size(long n, long bs_try) {
		assert(n > 0);
		long bs = bs_try;
		if (bs < 0) bs = n;
		if (bs > n) bs = n;
		return bs;
	}

	long num_blocks(long n, long bs) {
		assert(bs > 0);
		assert(n > 0);
		long nb = n/bs;
		if (nb*bs < n) nb++;
		assert(nb*bs >= n);
		assert((nb-1)*bs < n);
		return nb;
	}

	void update_ibeg_iend() {
		ibeg1 = ib1*bs1;
		ibeg2 = ib2*bs2;
		ibeg3 = ib3*bs3;

		iend1 = (ib1+1)*bs1;
		iend2 = (ib2+1)*bs2;
		iend3 = (ib3+1)*bs3;

		if (iend1 > base_slice.n1) iend1 = base_slice.n1;
		if (iend2 > base_slice.n2) iend2 = base_slice.n2;
		if (iend3 > base_slice.n3) iend3 = base_slice.n3;
	}
};

#endif
