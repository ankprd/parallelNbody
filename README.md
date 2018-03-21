This project presents different parallel and sequential algorithms for the nbody problem

Files list :
    test.sh : nice script for the demo
    set_env.sh : script from TD6 setting environment for CUDA
    folder mpi : different implementations of the bruteforce algorithm (sequential and MPI + OpenMP) and for the Barnes-Hut algorithm (sequential and MPI)
    folder cuda : different implementations of the bruteforce algorithm using CUDA and MPI

(There are two folders because of the complicated compilation for CUDA)

Before compiling anything, one should "source set_env.sh"

MPI FOLDER :

there are 4 different algorithms in this folder

Bruteforce sequential :
	File nbody_brute_force.c
	Compiled with "make nbody_brute_force"
	Executed with "salloc ./nbody_brute_force [nParticles [tMax]]"

Bruteforce parallel (MPI + OpenMP):
	File nbody_brute_force_parallel.c
	Compiled with "make nbody_brute_force_parallel"
	Executed with "salloc mpirun ./nbody_brute_force_parallel [nParticles [tMax]]"
	
Barnes-Hut sequential :
	File nbody_barnes_hut.c
	Compiled with "make nbody_barnes_hut"
	Executed with "salloc ./nbody_barnes_hut [nParticles [tMax]]"

Barnes-Hut parallel (MPI):
	File nbody_barnes_hut_parallel.c
	Compiled with "make nbody_barnes_hut_parallel"
	Executed with "salloc mpirun ./nbody_barnes_hut_parallel [nParticles [tMax]]"

	
CUDA FOLDER :
Warning : we could not get the Makefile to compile the nbody_alloc.c, nbody_tools.c, ui.c and xstuff.c files. Those are to be compiled in the MPI folder before the files in the cuda folder can be used.
The number of particles and time limit cannot be given as parameters when running the executables. They are to be changed in the cudaBruteforce.c, cudaMpi1Bruteforce.c or cudaMpi2Bruteforce.c

There are 3 different algorithms in this folder :

Bruteforce parallelization using only CUDA :
    Files cudaBruteforce.c and cudaFunc.cu
    Compiled with "make cudaS"
    
Bruteforce parallelization using CUDA and MPI v1 :
    Particles to calculate forces on are shared between MPI tasks and then between CUDA threads
    Files cudaMpi1Bruteforce.c and cudaMpi1Func.cu
    Compiled with "make cudaMpi1"

Bruteforce parallelization using CUDA and MPI v1 :
    Particles to calculate forces from are shared between MPI tasks and then particles to calculate forces on between CUDA threads
    Files cudaMpi2Bruteforce.c and cudaMpi2Func.cu
    Compiled with "make cudaMpi2"
