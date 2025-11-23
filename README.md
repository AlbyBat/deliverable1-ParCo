# deliverable1-ParCo
# Parallel Sparse Matrix-Vector Multiplication (SpMV) Benchmark

This project implements a parallel SpMV (Sparse Matrix-Vector Multiplication) algorithm in CSR format using OpenMP for shared-memory systems. The implementation includes performance benchmarking across different sparsity levels and scheduling strategies.



## Features

- **Parallel SpMV Implementation**: CSR format with OpenMP parallelization
- **Multiple Scheduling Strategies**: Static, dynamic, guided, and static-chunk
- **Comprehensive Benchmarking**: 5 sparsity levels (100k to 50M non-zero elements)
- **Strong Scaling Analysis**: 7 thread configurations (sequential to 32 threads)
- **Statistical Rigor**: 20 independent runs per configuration, 90th percentile reporting
- **Performance Metrics**: Time, GFLOPS, speedup, efficiency, bandwidth, arithmetic intensity

## Quick Start

## Local Execution (Linux) (Personal Computer)

Don't forget to change your machine theorical peak performance and total cores
### Compile with optimization and OpenMP support
g++ -O3 -fopenmp -o benchmark benchmark.cpp

### Run the benchmark
./benchmark

## HPC Execution

## 1. Login to HPC cluster
ssh username@hpc.unitn.it

## 2. Create project directory
mkdir deliverable1
cd deliverable1

## 3. Transfer files from local machine (run this on your local machine)
scp benchmark.cpp start.pbs username@hpc.unitn.it:~/deliverable1/

## 4. Prepare files (on HPC cluster)
dos2unix *.pbs *.cpp

## 5. Submit job to queue
qsub start.pbs

## 6. Monitor job status
qstat -u username

## 7. Check output after completion
cat csr_benchmark.out
