#ifndef DISTR_RNG_SPRNG2_H
#define DISTR_RNG_SPRNG2_H

// Parallel RNG using SPRNG2 library
// See http://www.sprng.org/Version2.0/users-guide.html

#include <cassert>

#define SIMPLE_SPRNG		/* simple interface                        */
#include <sprng_cpp.h>

#include "DomainDecomposition.hpp"
#include "Slice3d.hpp"

#define SEED 985456376

template <typename Dtype>
class DistrRngSprng2 {
	public:

	const DomainDecomposition & dd;
	long seed;

	int * my_stream;

	DistrRngSprng2(const DomainDecomposition & dd_, long seed_) : dd(dd_), seed(seed_) {
		// Init SPRNG2 stream for this rank
		// SPRNG2 streams are independent, see
		// http://www.sprng.org/Version2.0/definitions.html#independence
		// We use the combined multiple recursive generator, see
		// http://www.sprng.org/Version2.0/generators.html
//		my_stream = init_sprng(SPRNG_CMRG, dd.myrank, dd.np, seed, SPRNG_DEFAULT);
		my_stream = init_sprng(SEED, SPRNG_DEFAULT, DEFAULT_RNG_TYPE);
		if (dd.myrank == 0) print_sprng();
	}

	void fill_plusminusone(Slice3d<Dtype> & slice) const {
		// Fill the slice's cells with random +/- 1
		assert(slice.n == dd.ngp);
		const Dtype one = (Dtype) 1;

		for (long i3=0; i3<dd.ngp3; i3++) {
			for (long i2=0; i2<dd.ngp2; i2++) {
				for (long i1=0; i1<dd.ngp1; i1++) {
					double x = sprng();
					slice.element_ref(i1, i2, i3) = (x > 0.5 ? one : -one);
		} } }
	}

};

#endif
