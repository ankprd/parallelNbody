#!/bin/bash

CURDIR="$(pwd)"
PARADIR="/parallelMPI/"
PARADIR=$CURDIR$PARADIR
SEQDIR="/sequential/"
SEQDIR=$CURDIR$SEQDIR

cd $PARADIR
make nbody_brute_force
salloc mpirun nbody_brute_force

cd $SEQDIR
make nbody_brute_force
./nbody_brute_force

FILEN="particles.log"
FILES=$SEQDIR$FILEN
FILEP=$PARADIR$FILEN

comm -3 <(sort $FILES) <(sort $FILEN)
