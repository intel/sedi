CXX = mpiicpc -std=c++11 -O3 -xHost -Wall -pedantic -g -DALIGN -fopenmp

# Set defines for compile-time switches
DEFS = $(ADD_DEFS)
# Use linux-specific code, e.g. for memory reporting and FP exception setting
DEFS += -DWITH_LINUX_EXTENSIONS

SPRNG2_INC = 
SPRNG2_LIB = 

# Comment out to compile w/o HDF5 and data dump support
DEFS += -DWITH_HDF5
HDF5_INC = -I/$(HDF5_HOME)/include
HDF5_LIB = -L/$(HDF5_HOME)/lib -lhdf5

# Compile with support for MPI-3 nonblocking collectives
DEFS += -DWITH_MPI3

INCLUDES = $(SPRNG2_INC) $(HDF5_INC)
LIBS = $(SPRNG2_LIB) $(HDF5_LIB) -L/usr/lib64 -lgmp


main: stochastic-estimator.cg.exe stochastic-estimator.bicgstab.exe stochastic-estimator.pipelined_cg.exe

tests: test-border.exe test-halo.exe test-sprng.exe test-poisson.exe test-block-iterator.exe

all: main tests


stochastic-estimator.bicgstab.exe: *.hpp stochastic-estimator.cc
	$(CXX) $(DEFS) $(INCLUDES) -DSOLVER=1 -o $@ stochastic-estimator.cc $(LIBS)

stochastic-estimator.cg.exe: *.hpp stochastic-estimator.cc
	$(CXX) $(DEFS) $(INCLUDES) -DSOLVER=2 -o $@ stochastic-estimator.cc $(LIBS)

stochastic-estimator.pipelined_cg.exe: *.hpp stochastic-estimator.cc
	$(CXX) $(DEFS) $(INCLUDES) -DSOLVER=3 -o $@ stochastic-estimator.cc $(LIBS)


test-%.exe: *.hpp test-%.cc
	$(CXX) $(DEFS) $(INCLUDES) -o $@ test-$*.cc $(LIBS)


clean:
	rm -f *.exe
