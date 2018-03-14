#!/bin/bash

CURDIR="$(pwd)"
PARADIR="/parallelMPI/"
PARADIR=$CURDIR$PARADIR
SEQDIR="/sequential/"
SEQDIR=$CURDIR$SEQDIR

cd $PARADIR
make nbody_barnes_hut
salloc mpirun nbody_barnes_hut

cd $SEQDIR
make nbody_barnes_hut
./nbody_barnes_hut

FILEN="particles.log"
FILES=$SEQDIR$FILEN
FILEP=$PARADIR$FILEN

comm -3 <(sort $FILES) <(sort $FILEN)
