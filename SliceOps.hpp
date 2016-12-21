#ifndef SLICE_OPS_H
#define SLICE_OPS_H

#include "Slice3d.hpp"
#include "Timers.hpp"

// Helper functions to act on slices
template <typename Dtype>
class SliceOps {
	public:

	Timers * const timers;

	SliceOps(Timers * timers_ = 00) : timers(timers_) { }

	// z <- a*u + b*v
	void combine2(
			Dtype a, Slice3d<Dtype> & u,
			Dtype b, Slice3d<Dtype> & v,
			Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) =
						a*u.element_ref(i1, i2, i3) +
						b*v.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// z <- a*u + b*v + c*w
	void combine3(
			Dtype a, Slice3d<Dtype> & u,
			Dtype b, Slice3d<Dtype> & v,
			Dtype c, Slice3d<Dtype> & w,
			Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) =
						a*u.element_ref(i1, i2, i3) +
						b*v.element_ref(i1, i2, i3) +
						c*w.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// z <- u
	void copy(Slice3d<Dtype> & u, Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) = u.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// z <- u ./ v (elementwise division)
	void divide(Slice3d<Dtype> & u, Slice3d<Dtype> & v, Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) =
						u.element_ref(i1, i2, i3) / v.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// z <- u .* v (elementwise multiplication)
	void multiply(Slice3d<Dtype> & u, Slice3d<Dtype> & v, Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) =
						u.element_ref(i1, i2, i3) * v.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// z <- z + u .* v
	void accumulate_product(
			Slice3d<Dtype> & u, Slice3d<Dtype> & v, Slice3d<Dtype> & z) {

		if (timers) timers->start(Timers::TIMER_COMPUTE_MISC);
		for (long i3=0; i3<z.n3; i3++) {
			for (long i2=0; i2<z.n2; i2++) {
#pragma omp simd
				for (long i1=0; i1<z.n1; i1++) {
					z.element_ref(i1, i2, i3) +=
						u.element_ref(i1, i2, i3) * v.element_ref(i1, i2, i3);
		} } }
		if (timers) timers->stop(Timers::TIMER_COMPUTE_MISC);
	}

	// Dot product
	Dtype dotprod(const Slice3d<Dtype> & u, const Slice3d<Dtype> & v) {

		Dtype result = (Dtype)0;
		if (timers) timers->start(Timers::TIMER_COMPUTE_DOTPROD);
		for (long i3=0; i3<u.n3; i3++) {
			for (long i2=0; i2<u.n2; i2++) {
#pragma omp simd reduction(+:result)
				for (long i1=0; i1<u.n1; i1++) {
					result += u.element_ref(i1, i2, i3) * v.element_ref(i1, i2, i3);
				}
			}
		}
		if (timers) timers->stop(Timers::TIMER_COMPUTE_DOTPROD);
		return result;
	}

};

#endif
