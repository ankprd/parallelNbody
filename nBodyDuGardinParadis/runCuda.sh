#!/bin/bash
source set_env.sh

cd cuda

make clean
make all

echo ""
echo "bruteforce parallel with 10 000 particles with CUDA"
salloc ./cudaS

echo ""
echo "bruteforce parallel with 10 000 particles, 25 MPI nodes with CUDA"
salloc -N 25 --ntasks-per-node 1 mpirun ./cudaMpi2
