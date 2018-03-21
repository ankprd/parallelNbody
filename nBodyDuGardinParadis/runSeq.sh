#!/bin/bash

cd mpi

make clean
make nbody_brute_force

echo ""
echo "bruteforce sequential with 5 000 particles"
salloc ./nbody_brute_force 5000;
