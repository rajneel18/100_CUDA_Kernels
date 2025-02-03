#include <stdio.h>
#include <cuda.h>

#define N 1024 // Matrix size
#define TILE_SIZE 16 // Tile size for shared memory optimization

// Naïve Matrix Multiplication (No Tiling)
__global__ void matMulNaive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Optimized Matrix Multiplication (Using Tiling & Shared Memory)
__global__ void matMulTiled(float *A, float *B, float *C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0;
    
    for (int t = 0; t < n / TILE_SIZE; t++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // Synchronize threads before computation

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // Sync before loading next tile
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    // Allocate host memory
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_naive = (float*)malloc(bytes);
    float *h_C_tiled = (float*)malloc(bytes);

    // Initialize matrices with random values
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10);
        h_B[i] = (float)(rand() % 10);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C_naive, bytes);
    cudaMalloc(&d_C_tiled, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    // CUDA event setup for timing
    cudaEvent_t start, stop;
    float time_naive, time_tiled;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Run Naïve Kernel
    cudaEventRecord(start);
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_naive, start, stop);

    // Run Tiled Kernel
    cudaEventRecord(start);
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tiled, start, stop);

    // Copy results back to host
    cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost);

    // Print execution times
    printf("Naïve Matrix Multiplication Time: %.2f ms\n", time_naive);
    printf("Tiled Matrix Multiplication Time: %.2f ms\n", time_tiled);
    printf("Speedup: %.2fx\n", time_naive / time_tiled);

    // Free memory
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);

    return 0;
}
