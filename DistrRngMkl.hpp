#ifndef DISTR_RNG_MKL_H
#define DISTR_RNG_MKL_H

// Use MKL's block-splitting RNG functionality to generate independent streams
// on each MPI rank. See:
//
//   https://software.intel.com/en-us/node/521867
//
// Note that using different seeds on different MPI ranks does NOT guarantee
// that the generated variates are independent, see:
//
//   https://software.intel.com/en-us/forums/intel-math-kernel-library/topic/283349
//

#include <mkl_vsl.h>
#include "DomainDecomposition.hpp"

class DistrRngMkl {
	private:
	VSLStreamStatePtr the_stream, my_stream;

	public:
	DistrRngMklnst DomainDecomp & dd, long seed) {
		// Initialize the_stream to a SAME stream RNG on all MPI ranks (fixed seed)
		vslNewStream(&the_stream, VSL_BRNG_MCG31, seed);

		// Compute my own skip
		long my_skip = dd.my_rank * dd.ncells_per_rank;

		// Make my_stream by shifting the_stream
		vslCopyStream(&my_stream, the_stream);
		int status = vslSkipAheadStream(my_stream, my_skip);
		assert(status == VSL_STATUS_OK);
	}

	~DistrRngMkl{
		vslDeleteStream(&my_stream);
		vslDeleteStream(&the_stream);
	}
}

#endif
