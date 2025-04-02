#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(error) { \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for LoRA matrix multiplication
__global__ void lora_kernel(const float* x, const float* W, const float* A, const float* B, float* y,
                            int M, int N, int K, int R) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        float acc = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float sum_ab = 0.0f;
           
            for (int r = 0; r < R; ++r) {
                sum_ab += A[k * R + r] * B[r * N + col];
            }
           
            float w_eff = W[k * N + col] + sum_ab;
            acc += x[row * K + k] * w_eff;
        }
        y[row * N + col] = acc;
    }
}

// CPU function for LoRA matrix multiplication
void cpu_lora(const float* x, const float* W, const float* A, const float* B, float* y,
              int M, int N, int K, int R) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float sum_ab = 0.0f;
                for (int r = 0; r < R; r++) {
                    sum_ab += A[k * R + r] * B[r * N + j];
                }
                float w_eff = W[k * N + j] + sum_ab;
                acc += x[i * K + k] * w_eff;
            }
            y[i * N + j] = acc;
        }
    }
}

// Function to print a portion of a matrix
void print_matrix(const float* mat, int M, int N, int max_rows, int max_cols) {
    for (int i = 0; i < M && i < max_rows; ++i) {
        for (int j = 0; j < N && j < max_cols; ++j) {
            printf("%.5f\t", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    // Define matrix dimensions
    int M = 128, K = 256, N = 64, R = 32;
    
    // Allocate memory for host matrices
    size_t size_x = M * K * sizeof(float);
    size_t size_W = K * N * sizeof(float);
    size_t size_A = K * R * sizeof(float);
    size_t size_B = R * N * sizeof(float);
    size_t size_y = M * N * sizeof(float);

    float *h_x = (float*)malloc(size_x);
    float *h_W = (float*)malloc(size_W);
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_y = (float*)malloc(size_y);
    float *h_y_cpu = (float*)malloc(size_y);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) h_x[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_W[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * R; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < R * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Allocate memory for device matrices
    float *d_x, *d_W, *d_A, *d_B, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void**)&d_W, size_W));
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_y, size_y));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Define grid and block dimensions
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    // Record start time
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch CUDA kernel
    lora_kernel<<<blocks, threads>>>(d_x, d_W, d_A, d_B, d_y, M, N, K, R);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure execution time
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost));

    // CPU execution
    clock_t cpu_start = clock();
    cpu_lora(h_x, h_W, h_A, h_B, h_y_cpu, M, N, K, R);
    clock_t cpu_end = clock();
    float cpuTime = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Compare CPU and GPU results
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_y_cpu[i] - h_y[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    printf("Max difference between CPU and GPU results: %f\n", max_diff);
    printf("GPU kernel execution time: %f ms\n", gpuTime);
    printf("CPU execution time: %f ms\n", cpuTime);

    // Print sample output
    printf("\nSample output from GPU result (first 5 rows, 5 cols):\n");
    print_matrix(h_y, M, N, 5, 5);

    printf("\nSample output from CPU result (first 5 rows, 5 cols):\n");
    print_matrix(h_y_cpu, M, N, 5, 5);

    // Clean up resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_x); free(h_W); free(h_A); free(h_B); free(h_y); free(h_y_cpu);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_A); cudaFree(d_B); cudaFree(d_y);

    return 0;
}
