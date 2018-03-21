export OMPI_CC=$(shell echo "$(CUDA_ROOT)")/bin/nvcc

export OMPI_CFLAGS=-I/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/include

export OMPI_LDFLAGS=-I/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/include -L/users/profs/2017/francois.trahay/soft/install/hwloc/lib -Xlinker=-rpath -Xlinker=/users/profs/2017/francois.trahay/soft/install/hwloc/lib -Xlinker=-rpath -Xlinker=/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/lib -Xlinker=--enable-new-dtags -L/users/profs/2017/francois.trahay/soft/install/openmpi-2.1.2/lib -lmpi -lX11

export VAR=truccoucou

OBJS = ui.o xstuff.o nbody_tools.o nbody_alloc.o

HEADERS=.
CFLAGS = -I$(HEADERS)

main.o:main.c
	mpicc -c -arch sm_30 -O3 main.c $@ $< $(OBJS) $(CFLAGS)

hello_cpu.o:hello_cpu.cu
	mpicc -c -arch sm_30 -O3 hello_cpu.cu $(CFLAGS)

cross:
	mpicc main.o hello_cpu.o $(OBJS) -o cross -lcudart

env:
	echo $$VAR

debug:
	echo $$OMPI_CC
	echo $$OMPI_CFLAGS
	echo $$OMPI_LDFLAGS
