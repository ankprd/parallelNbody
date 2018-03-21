export OMPI_CC=$(shell echo "$(CUDA_ROOT)")/bin/nvcc

export OMPI_CFLAGS=-I/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/include

export OMPI_LDFLAGS=-I/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/include -L/users/profs/2017/francois.trahay/soft/install/hwloc/lib -Xlinker=-rpath -Xlinker=/users/profs/2017/francois.trahay/soft/install/hwloc/lib -Xlinker=-rpath -Xlinker=/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/lib -Xlinker=--enable-new-dtags -L/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/lib -lmpi -lX11

OBJS = "../mpi/ui.o" "../mpi/xstuff.o" "../mpi/nbody_tools.o" "../mpi/nbody_alloc.o"

HEADERS=.
CFLAGS = -I$(HEADERS)

all: cudaS cudaMpi1 cudaMpi2 

cudaBruteforce.o:cudaBruteforce.c
	mpicc -c -arch sm_30 -O3 cudaBruteforce.c $@ $< $(OBJS) $(CFLAGS)

cudaFunc.o:cudaFunc.cu
	mpicc -c -arch sm_30 -O3 cudaFunc.cu $(CFLAGS)

cudaS: cudaBruteforce.o cudaFunc.o
	mpicc cudaBruteforce.o cudaFunc.o $(OBJS) -o cudaS -lcudart
        
cudaMpi1Bruteforce.o : cudaMpi1Bruteforce.c
	mpicc -c -arch sm_30 -O3 cudaMpi1Bruteforce.c $@ $< $(OBJS) $(CFLAGS)

cudaMpi1Func.o:cudaMpi1Func.cu
	mpicc -c -arch sm_30 -O3 cudaMpi1Func.cu $(CFLAGS)

cudaMpi1: cudaMpi1Bruteforce.o cudaMpi1Func.o
	mpicc cudaMpi1Bruteforce.o cudaMpi1Func.o $(OBJS) -o cudaMpi1 -lcudart

cudaMpi2Bruteforce.o : cudaMpi2Bruteforce.c
	mpicc -c -arch sm_30 -O3 cudaMpi2Bruteforce.c $@ $< $(OBJS) $(CFLAGS)

cudaMpi2Func.o:cudaMpi2Func.cu
	mpicc -c -arch sm_30 -O3 cudaMpi2Func.cu $(CFLAGS)

cudaMpi2: cudaMpi2Bruteforce.o cudaMpi2Func.o
	mpicc cudaMpi2Bruteforce.o cudaMpi2Func.o $(OBJS) -o cudaMpi2 -lcudart

clean:
	rm -f cudaMpi2Bruteforce.o cudaMpi2Func.o cudaMpi2 cudaBruteforce.o cudaFunc.o cudaS cudaMpi1Bruteforce.o cudaMpi1Func.o cudaMpi1

debug:
	echo $$OMPI_CC
	echo $$OMPI_CFLAGS
	echo $$OMPI_LDFLAGS
