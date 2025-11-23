#!/bin/bash
#PBS -N csr_benchmark
#PBS -o ./csr_benchmark.out
#PBS -e ./csr_benchmark.err
#PBS -q short_cpuQ
#PBS -l walltime=0:10:00
#PBS -l select=1:ncpus=32:mem=50gb

cd deliverable1

module load gcc91

g++ -fopenmp -O3 -o benchmark benchmark.cpp

export OMP_NUM_THREADS=32
./benchmark
