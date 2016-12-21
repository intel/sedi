#ifndef STENCIL_H
#define STENCIL_H

#include <cassert>
#include "Slice3d.hpp"
#include "BlockSliceIterator.hpp"
#include "Platform.hpp"

template <typename Dtype>
class Stencil {
	public:
	Dtype a, b, c;

#ifndef DISABLE_CACHE_BLOCKING
#define DISABLE_CACHE_BLOCKING 0
#endif

	Stencil(Dtype a_, Dtype b_, Dtype c_) :
		a(a_), b(b_), c(c_) { }

	void apply(const Slice3d<Dtype> & x, Slice3d<Dtype> & y) const {
		assert( x.n1 == y.n1 && x.n2 == y.n2 && x.n3 == y.n3 );

		if (DISABLE_CACHE_BLOCKING || 2*x.n*sizeof(Dtype)*2 < PLATFORM_L3_BYTES_PER_CORE) {
			// Don't perform cache blocking
			apply_direct(x, y);
		} else {
			// Iterate by cache blocks
			BlockSliceIterator<Dtype> iter = BlockSliceIterator<Dtype>::cache_aware_iterator(x, PLATFORM_L3_BYTES_PER_CORE, 4);
			for (; !iter.end(); iter.next()) {
				Slice3d<Dtype> x_block = iter.get_block(x);
				Slice3d<Dtype> y_block = iter.get_block(y);
				apply_direct(x_block, y_block);
			}
		}
	}

	void apply_direct(const Slice3d<Dtype> & x, Slice3d<Dtype> & y) const {
		for (long i3=0; i3<x.n3; i3++) {
			for (long i2=0; i2<x.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<x.n1; i1++) {
					y.element_ref(i1, i2, i3) =
						a * x.element_ref(i1, i2, i3)
						- c * x.element_ref(i1-1, i2, i3)
						- c * x.element_ref(i1, i2-1, i3)
						- c * x.element_ref(i1, i2, i3-1)
						- b * x.element_ref(i1+1, i2, i3)
						- b * x.element_ref(i1, i2+1, i3)
						- b * x.element_ref(i1, i2, i3+1);
				}
			}
		}
	}

};

#endif
