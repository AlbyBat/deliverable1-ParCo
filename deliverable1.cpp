#include <pthread.h>
#include <semaphore.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define NUM_RIGHE 50000
#define NUM_COLONNE 25000
#define NUM_RUNS 20
#define NUM_MIN_RAND -10.0
#define NUM_MAX_RAND 10.0

//5 levels of sparsity
int sparse_levels[] = {100000, 500000, 2500000, 10000000, 50000000};
const char* sparse_names[] = {"Extreme-Sparse", "Very-Sparse", "Sparse", "Moderate", "Dense"};

//4 types of scheduling
typedef enum {
    SCHED_STATIC,
    SCHED_DYNAMIC,
    SCHED_GUIDED,
    SCHED_STATIC_CHUNK
} ScheduleType;
const char* schedule_names[] = {"Static", "Dynamic", "Guided", "Static-Chunk"};

int num_sparse_levels = sizeof(sparse_levels) / sizeof(sparse_levels[0]);

int thread_configs[] = {0, 1, 2, 4, 8, 16, 32};  //0 =sequential, no openmp
int num_thread_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);

//spec hpc
const double PEAK_TFLOPS = 2.976;           //theoretical peak performance
const double PEAK_GFLOPS = PEAK_TFLOPS * 1000.0;
const int TOTAL_CORES = 32;    //number of cores
const double PEAK_GBs = 100.0;              //initial value that will be overwritten
const double RIDGE_POINT = PEAK_GFLOPS / PEAK_GBs;

typedef struct {
    double *val;
    int *col_ind;
    int *row_ptr;
    int nnz;
    int rows;
    int cols;
} CSRMatrix;

double *matrice = NULL;
double *x = NULL;
CSRMatrix csr = {NULL, NULL, NULL, 0, 0, 0};

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double randValore(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void init_matrix(int nnz_target) {
    if (matrice != NULL) free(matrice);
    matrice = (double*)malloc(NUM_RIGHE * NUM_COLONNE * sizeof(double));
    
    //init with zeros
    for(int i = 0; i < NUM_RIGHE * NUM_COLONNE; i++) {
        matrice[i] = 0.0;
    }
    
    //add nonzero values
    int non_zero_count = 0;
    while (non_zero_count < nnz_target) {
        int idx = rand() % (NUM_RIGHE * NUM_COLONNE);
        if (matrice[idx] == 0.0) {
            matrice[idx] = randValore(NUM_MIN_RAND, NUM_MAX_RAND);
            non_zero_count++;
        }
    }
}

void init_csr() {
    //count nonzero elements ... will be the same as sparse_levels[n]
    int nnz_actual = 0;
    for(int i = 0; i < NUM_RIGHE * NUM_COLONNE; i++) {
        if(matrice[i] != 0.0) nnz_actual++;
    }
    
    //free previous allocations
    if (csr.val != NULL) free(csr.val);
    if (csr.col_ind != NULL) free(csr.col_ind);
    if (csr.row_ptr != NULL) free(csr.row_ptr);
    
    //allocate csr structure
    csr.val = (double*)malloc(nnz_actual * sizeof(double));
    csr.col_ind = (int*)malloc(nnz_actual * sizeof(int));
    csr.row_ptr = (int*)malloc((NUM_RIGHE + 1) * sizeof(int));
    csr.nnz = nnz_actual;
    csr.rows = NUM_RIGHE;
    csr.cols = NUM_COLONNE;
    
    //fill csr
    int count = 0;
    csr.row_ptr[0] = 0;
    
    for(int i = 0; i < NUM_RIGHE; i++) {
        for(int j = 0; j < NUM_COLONNE; j++) {
            double val = matrice[i * NUM_COLONNE + j];
            if(val != 0.0) {
                csr.val[count] = val;
                csr.col_ind[count] = j;
                count++;
            }
        }
        csr.row_ptr[i+1] = count;
    }
}

void init_vector() {
    if (x != NULL) free(x);
    x = (double*)malloc(NUM_COLONNE * sizeof(double));
    for(int i = 0; i < NUM_COLONNE; i++) {
        x[i] = randValore(NUM_MIN_RAND, NUM_MAX_RAND);
    }
}

double calculate_operational_intensity() {
    //flops for SPMV: 2 operations per nonzero (1 multiply + 1 add)
    double total_flops = 2.0 * csr.nnz;
    
    //menory traffic estimation for CSR SpMV:
    //read: csr.val[], csr.col_ind[], x[] (accessed via col_ind)
    //write: result[] (one per row)
    double total_bytes = (csr.nnz * sizeof(double)) +      //csr.val
                        (csr.nnz * sizeof(int)) +         //csr.col_ind  
                        (csr.nnz * sizeof(double)) +      //x[] accesse
                        (csr.rows * sizeof(double));      //result[]
    
    return total_flops / total_bytes;
}

double calculate_gflops(double time_seconds) {
    if (time_seconds <= 0) return 0.0;
    double total_flops = 2.0 * csr.nnz;
    return (total_flops / time_seconds) / 1e9;
}

double estimate_bandwidth_from_spmv(double time_seconds) {
    if (time_seconds <= 0) return 0.0;
    
    double total_bytes = (csr.nnz * sizeof(double)) +      //csr.val
                        (csr.nnz * sizeof(int)) +         //csr.col_ind
                        (csr.nnz * sizeof(double)) +      //x[] accesse
                        (csr.rows * sizeof(double));      //result[]
    
    return (total_bytes / time_seconds) / 1e9; //to gb/s
}

void analyze_bottleneck(double measured_time_seconds, double empirical_bandwidth) {
    double total_flops = 2.0 * csr.nnz;
    double total_bytes = (csr.nnz * sizeof(double)) + 
                        (csr.nnz * sizeof(int)) +
                        (csr.nnz * sizeof(double)) + 
                        (csr.rows * sizeof(double));
    
    double T_mem = total_bytes / (empirical_bandwidth * 1e9);
    double T_math = total_flops / (PEAK_GFLOPS * 1e9);
    
    double arithmetic_intensity = calculate_operational_intensity();
    double ops_byte_ratio = PEAK_GFLOPS / empirical_bandwidth;
    
    printf("\n--- BOTTLENECK ANALYSIS ---\n");
    printf("Total FLOPs: %.0f\n", total_flops);
    printf("Total Bytes: %.0f\n", total_bytes);
    printf("Arithmetic Intensity: %.6f FLOPs/byte\n", arithmetic_intensity);
    printf("Processor's Ops/Byte Ratio: %.6f FLOPs/byte\n", ops_byte_ratio);
    printf("Empirical Bandwidth: %.2f GB/s\n", empirical_bandwidth);
    printf("T_mem (Memory Time): %.9f seconds\n", T_mem);
    printf("T_math (Compute Time): %.9f seconds\n", T_math);
    printf("Measured Time: %.9f seconds\n", measured_time_seconds);
    
    if (T_mem > T_math) {
        printf(" ALGORITHM IS MEMORY-BOUND (T_mem > T_math)\n");
    } else {
        printf(" ALGORITHM IS COMPUTE-BOUND (T_math > T_mem)\n");
    }
    
    if (arithmetic_intensity < ops_byte_ratio) {
        printf(" Arithmetic Intensity < Processor's Ops/Byte Ratio\n");
        printf("   %.6f < %.6f\n", arithmetic_intensity, ops_byte_ratio);
    } else {
        printf(" Arithmetic Intensity >= Processor's Ops/Byte Ratio\n");
        printf("   %.6f >= %.6f\n", arithmetic_intensity, ops_byte_ratio);
    }
    
    //calcualte efficiency related to theorical peak
    double achieved_gflops = calculate_gflops(measured_time_seconds);
    double efficiency = (achieved_gflops / PEAK_GFLOPS) * 100.0;
    printf("Achieved Performance: %.2f GFLOPS (%.1f%% of peak)\n", achieved_gflops, efficiency);
}

//sequential version
void multiply_sequential(double *result) {
    for(int i = 0; i < csr.rows; i++) {
        result[i] = 0.0;
        for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
            result[i] += csr.val[k] * x[csr.col_ind[k]];
        }
    }
}

//parallel version with openmp (base)
void multiply_parallel(double *result, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(int i = 0; i < csr.rows; i++) {
        double sum = 0.0;
        for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
            sum += csr.val[k] * x[csr.col_ind[k]];
        }
        result[i] = sum;
    }
}

//parallel version with different scheduling strategies
void multiply_parallel_v2(double *result, int num_threads, ScheduleType sched_type) {
    #pragma omp parallel num_threads(num_threads)
    {
        switch(sched_type) {
            case SCHED_STATIC:
                #pragma omp for schedule(static)
                for(int i = 0; i < csr.rows; i++) {
                    double sum = 0.0;
                    for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
                        sum += csr.val[k] * x[csr.col_ind[k]];
                    }
                    result[i] = sum;
                }
                break;
                
            case SCHED_DYNAMIC:
                #pragma omp for schedule(dynamic, 16)
                for(int i = 0; i < csr.rows; i++) {
                    double sum = 0.0;
                    for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
                        sum += csr.val[k] * x[csr.col_ind[k]];
                    }
                    result[i] = sum;
                }
                break;
                
            case SCHED_GUIDED:
                #pragma omp for schedule(guided)
                for(int i = 0; i < csr.rows; i++) {
                    double sum = 0.0;
                    for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
                        sum += csr.val[k] * x[csr.col_ind[k]];
                    }
                    result[i] = sum;
                }
                break;
                
            case SCHED_STATIC_CHUNK:
                #pragma omp for schedule(static, 64)
                for(int i = 0; i < csr.rows; i++) {
                    double sum = 0.0;
                    for(int k = csr.row_ptr[i]; k < csr.row_ptr[i+1]; k++) {
                        sum += csr.val[k] * x[csr.col_ind[k]];
                    }
                    result[i] = sum;
                }
                break;
        }
    }
}

//function to calculate 90th percentile
double percentile_90(double *data, int n) {
    std::vector<double> sorted(data, data + n);
    std::sort(sorted.begin(), sorted.end());
    return sorted[(int)(n * 0.9)];
}

void benchmark_with_different_schedules(int sparsity_index) {
    int nnz_target = sparse_levels[sparsity_index];
    init_matrix(nnz_target);
    init_csr();
    init_vector();
    
    printf("\nTesting different scheduling strategies for %s matrix:\n", 
           sparse_names[sparsity_index]);
    
    for(int sched = 0; sched <= SCHED_STATIC_CHUNK; sched++) {
        double times[NUM_RUNS];
        
        for(int run = 0; run < NUM_RUNS; run++) {
            double *result = (double*)malloc(csr.rows * sizeof(double));
            double start_time = get_time();
            
            multiply_parallel_v2(result, 32, (ScheduleType)sched);
            
            double end_time = get_time();
            times[run] = (end_time - start_time) * 1000;
            free(result);
        }
        
        double time_ms = percentile_90(times, NUM_RUNS);
        printf("%s: %.2f ms\n", schedule_names[sched], time_ms);
    }
}

void benchmark_sparsity_level(int sparsity_index) {
    int nnz_target = sparse_levels[sparsity_index];
    const char* sparsity_name = sparse_names[sparsity_index];
    
    printf("\n=== %s Matrix (%d non-zero elements) ===\n", sparsity_name, nnz_target);
    
    init_matrix(nnz_target);
    init_csr();
    init_vector();
    
    double sparsity_percent = 100.0 * (1.0 - (double)csr.nnz/(NUM_RIGHE * NUM_COLONNE));
    double operational_intensity = calculate_operational_intensity();
    
    printf("Matrix: %d x %d, Actual non-zero: %d, Sparsity: %.2f%%\n",
           NUM_RIGHE, NUM_COLONNE, csr.nnz, sparsity_percent);
    printf("Operational Intensity: %.6f FLOPs/byte\n", operational_intensity);
    
    if(operational_intensity < RIDGE_POINT) {
        printf("Theoretical Prediction: MEMORY-BOUND\n");
    } else {
        printf("Theoretical Prediction: COMPUTE-BOUND\n");
    }
    
    printf("\nThreads\tTime(ms)\tGFLOPS\t\tEff(%%) \tSpeedup\tBandwidth(GB/s)\n");
    printf("-------\t--------\t-------\t\t------\t-------\t---------------\n");
    
    double sequential_time = 0.0;
    double best_empirical_bandwidth = 0.0;
    
    for(int config = 0; config < num_thread_configs; config++) {
        int threads = thread_configs[config];
        double times[NUM_RUNS];
        double gflops_rates[NUM_RUNS];
        double bandwidths[NUM_RUNS];
        
        for(int run = 0; run < NUM_RUNS; run++) {
            double *result = (double*)malloc(csr.rows * sizeof(double));
            double start_time = get_time();
            
            if(threads == 0) {
                multiply_sequential(result);
            } else {
                multiply_parallel(result, threads);
            }
            
            double end_time = get_time();
            double elapsed_time = (end_time - start_time);
            
            times[run] = elapsed_time * 1000; //ms
            gflops_rates[run] = calculate_gflops(elapsed_time);
            bandwidths[run] = estimate_bandwidth_from_spmv(elapsed_time);
            
            free(result);
        }
        
        double time_ms = percentile_90(times, NUM_RUNS);
        double gflops = percentile_90(gflops_rates, NUM_RUNS);
        double bandwidth = percentile_90(bandwidths, NUM_RUNS);
        
        if(bandwidth > best_empirical_bandwidth) {
            best_empirical_bandwidth = bandwidth;
        }
        
        if(threads == 0) {
            sequential_time = time_ms;
        }
        
        double speedup = (threads == 0) ? 1.0 : sequential_time / time_ms;
        double efficiency = (threads == 0) ? 100.0 : (speedup / threads) * 100;
        
        if(threads == 0) {
            printf("Seq\t%.2f\t\t%.2f\t\t%.1f\t%.2f\t%.2f\n",
                   time_ms, gflops, efficiency, speedup, bandwidth);
        } else {
            printf("%d\t%.2f\t\t%.2f\t\t%.1f\t%.2f\t%.2f\n",
                   threads, time_ms, gflops, efficiency, speedup, bandwidth);
        }
    }
    
    analyze_bottleneck(sequential_time / 1000.0, best_empirical_bandwidth);
}

void strong_scaling_benchmark() {
    printf("=== STRONG SCALING ANALYSIS ===\n");
    printf("Fixed Problem Size: %d x %d matrix\n", NUM_RIGHE, NUM_COLONNE);
    printf("Hardware Specifications:\n");
    printf("  - Theoretical Peak: %.1f TFLOPS (%.1f GFLOPS)\n", PEAK_TFLOPS, PEAK_GFLOPS);
    printf("  - Number of Cores: %d\n", TOTAL_CORES);
    printf("  - Ridge Point: %.6f FLOPs/byte\n", RIDGE_POINT);
    printf("Number of runs per configuration: %d\n", NUM_RUNS);
    printf("Thread configurations: ");
    for(int i = 0; i < num_thread_configs; i++) {
        if(thread_configs[i] == 0) {
            printf("Sequential");
        } else {
            printf("%d", thread_configs[i]);
        }
        if(i < num_thread_configs - 1) printf(", ");
    }
    printf("\n\n");
    
    for(int s = 0; s < num_sparse_levels; s++) {
        benchmark_sparsity_level(s);
    }
}

void roofline_analysis() {
    printf("\n=== ROOFLINE MODEL ANALYSIS ===\n");
    printf("Using empirical bandwidth measurements from SpMV benchmark\n");
    
    init_matrix(sparse_levels[4]);
    init_csr();
    init_vector();
    
    double *result = (double*)malloc(csr.rows * sizeof(double));
    double start_time = get_time();
    multiply_sequential(result);
    double end_time = get_time();
    double elapsed = end_time - start_time;
    
    double empirical_bandwidth = estimate_bandwidth_from_spmv(elapsed);
    free(result);
    
    printf("Measured Bandwidth: %.2f GB/s\n", empirical_bandwidth);
    
    printf("\nRoofline Curve Points (using empirical bandwidth):\n");
    printf("Operational_Intensity, Performance_GFLOPS\n");
    
    //roofline curve points 
    for(double oi = 0.001; oi < 10.0; oi *= 2.0) {
        double performance = fmin(PEAK_GFLOPS, oi * empirical_bandwidth);
        printf("%.6f, %.2f\n", oi, performance);
    }
    
    printf("\nApplication Points:\n");
    printf("Sparsity_Level, Operational_Intensity, Performance_GFLOPS, Bound_Type\n");
    
    for(int s = 0; s < num_sparse_levels; s++) {
        init_matrix(sparse_levels[s]);
        init_csr();
        init_vector(); 
        
        double operational_intensity = calculate_operational_intensity();
        double attainable_performance = fmin(PEAK_GFLOPS, operational_intensity * empirical_bandwidth);
        const char* bound_type = (operational_intensity < (PEAK_GFLOPS / empirical_bandwidth)) ? "Memory" : "Compute";
        
        printf("%s, %.6f, %.2f, %s\n", 
               sparse_names[s], operational_intensity, attainable_performance, bound_type);
        
        free(matrice);
        free(csr.val);
        free(csr.col_ind);
        free(csr.row_ptr);
        free(x);
        matrice = NULL;
        csr.val = NULL;
        csr.col_ind = NULL;
        csr.row_ptr = NULL;
        x = NULL;
    }
}

int main() {
    srand(time(NULL));
    
    printf("=== CSR Matrix-Vector Multiplication Benchmark ===\n");
    printf("Strong Scaling Analysis with Empirical Bandwidth Measurement\n");
    printf("Matrix Dimensions: %d x %d\n", NUM_RIGHE, NUM_COLONNE);
    printf("HPC Specifications:\n");
    printf("  - Theoretical Peak: %.1f TFLOPS\n", PEAK_TFLOPS);
    printf("  - CPU Cores: %d\n", TOTAL_CORES);
    printf("  - Initial Bandwidth Estimate: %.1f GB/s\n", PEAK_GBs);
    
    strong_scaling_benchmark();
    
    roofline_analysis();
    
    printf("\n=== SCHEDULING STRATEGIES COMPARISON ===\n");
    for(int s = 0; s < num_sparse_levels; s++) {
        benchmark_with_different_schedules(s);
    }
    
    if (matrice != NULL) free(matrice);
    if (x != NULL) free(x);
    if (csr.val != NULL) free(csr.val);
    if (csr.col_ind != NULL) free(csr.col_ind);
    if (csr.row_ptr != NULL) free(csr.row_ptr);
    
    printf("\n=== BENCHMARK COMPLETED ===\n");
    
    return 0;
}
