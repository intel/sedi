# sedi

Welcome to the Stochastic Estimation of the Inverse of the Diagonal (SEDI) program.

PURPOSE
=========

The goal of this application is to compute the diagonal elements of the inverse of a large matrix 
using a stochastic approach. As such, it does not compute the real values of the diagonal elements 
of the inverse of the matrix but it provides an accurate estimation of these values.
The application defines an operator on a 3D cartesian grid that acts as a stencil. It implements 
a stochastic estimator technique to compute the diagonal of the inverse of this operator.
It involves distributed memory stencil computations (so a neighbor halo exchange), and a CG algorithm
(distributed dot products, norms, ...) that can be either a regular CG, a pipelined CG or a BiCGStab.

This code is written in C++. It is parallel using the MPI standard and it is designed to scale up
to large numbers of MPI ranks. 


COMPILATION
=============

A Makefile is provided with the source code. Simply type `make` in the directory where you installed 
the source files in order to generate the 3 following executables:
- stochastic-estimator.cg.exe
- stochastic-estimator.pipelined_cg.exe
- stochastic-estimator.bicgstab.exe 

What you need to do:
- provide a C++ compiler and its desired optimization options
- provide a MPI library (either through a wrapper of the C++ compiler or explicitely by giving the 
appropriate include and library paths (in flags INCLUDE and LIBS respectively).

SEDI provides with the ability to dump output results in HDF5 format. This is optional.
If this option is set in the makefile, please, provide the adequatte include and library paths for
flag HDF5_INC and HDF5_LIB respectively.


RANDOM NUMBER GENERATOR
=========================

The proof of concept of SEDI requires solving a large number of linear solvers with random right hand sides.
A random number generator engine is thus required.

Simple SEDI implementation makes use of the C++11 Mersenne Twister Algorithm (see the DistrRngStd.hpp class).
Other random number generator engines may be used. Examples are provided within SEDI to switch from the regular
C++11 RNG to:

	- Use MKL's block-splitting RNG functionality to generate independent streams
	- Provide parallel RNG using SPRNG2/SPRNG5 libraries
	- Provide parallel RNG using the BOOST library

Note: No support is provided for these third party libraries. Related C++ classes examples are for educational 
purpose only. Modifications in the caller function may be required.


EXECUTION
===========

In order to launch an execution of SEDI, here is a list of parameters that can be used:
`sedi_progname.exe` -g NxNxN -p PxPxP [-s NSAMPLES] [-e TOL] [-m MAXITER] [-S] [-i]

where:

	-g NxNxN : size of the global 3D grid
	-p PxPxP : decomposition into ranks
	-s : number of samples for stochastic estimation
	-e : solver tolerance
	-m : solver maximum iterations
	-i : ignore solver convergence errors and keep going


Examples of command line execution for a 128^3 grid on 64 MPI ranks:

1. Fixed numbers of samples (eg. 5) and iterations (eg. 50)

`mpirun -n 64 stochastic-estimator.cg.exe -g 128x128x128 -p 4x4x4 -s 5 -e 1e-9 -m 50 -i	`

2. Fixed number of samples and run to convergence

`mpirun -n 64 stochastic-estimator.cg.exe -g 128x128x128 -p 4x4x4 -s 5 -i`

