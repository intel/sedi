#ifndef DISTR_RNG_STD_H
#define DISTR_RNG_STD_H

// Parallel RNG using C++11 Mersenne Twister Algorithm

#include <cassert>

#include <random>

#include "DomainDecomposition.hpp"
#include "Slice3d.hpp"


template <typename Dtype>
class DistrRngStd {
	public:

	const DomainDecomposition & dd;

	DistrRngStd(const DomainDecomposition & dd_) : dd(dd_) { }

	void fill_plusminusone(Slice3d<Dtype> & slice) const {
		// Fill the slice's cells with random +/- 1
		assert(slice.n == dd.ngp);
		const Dtype one = (Dtype) 1;

		std::random_device rd;
		// Use Mersenne Twister (MT) engine to generate pseudo-random numbers.
		std::mt19937 engine(rd());
		// Filter MT engine's output to generate pseudo-random double values,
		// uniformly distributed on the closed interval [0, 1].
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		for (long i3=0; i3<dd.ngp3; i3++) {
			for (long i2=0; i2<dd.ngp2; i2++) {
				for (long i1=0; i1<dd.ngp1; i1++) {
					// Generate pseudo-random number.
					double x = dist(engine);
					// Fill slice with associated random +/- 1
					slice.element_ref(i1, i2, i3) = (x > 0.5 ? one : -one);
		} } }
	}

};

#endif
