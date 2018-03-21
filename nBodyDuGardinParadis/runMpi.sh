#!/bin/bash

export OMP_NUM_THREADS=8

cd mpi

make clean
make

echo ""
echo "bruteforce parallel with 10 000 particles, 32 MPI tasks and 8 OpenMP tasks"
salloc -n 32 -c 8 mpirun ./nbody_brute_force_parallel 10000;
echo ""
echo "barnes-hut sequential with 10 000 particles"
salloc ./nbody_barnes_hut 10000;
echo ""
echo "barnes-hut parallel with 10 000 particles and 32 MPI tasks"
salloc -n 32 -N 8 ./nbody_barnes_hut_parallel 10000;

cd ../
