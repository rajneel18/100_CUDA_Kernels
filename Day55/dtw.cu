#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  
#define M 1024  
#define BLOCK_SIZE 16  

__device__ float absDiff(float a, float b) {
    return fabsf(a - b);
}

// Kernel to compute DTW cost matrix in parallel
__global__ void computeDTWCostMatrix(float *d_X, float *d_Y, float *d_Cost, int n, int m) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < n && j < m) {
        float cost = absDiff(d_X[i], d_Y[j]);

        if (i == 0 && j == 0) {
            d_Cost[i * m + j] = cost;
        } else if (i == 0) {
            d_Cost[i * m + j] = cost + d_Cost[i * m + (j - 1)];
        } else if (j == 0) {
            d_Cost[i * m + j] = cost + d_Cost[(i - 1) * m + j];
        } else {
            float minCost = fminf(fminf(d_Cost[(i - 1) * m + j], 
                                       d_Cost[i * m + (j - 1)]), 
                                       d_Cost[(i - 1) * m + (j - 1)]);
            d_Cost[i * m + j] = cost + minCost;
        }
    }
}

int main() {

    float *h_X = (float*)malloc(N * sizeof(float));
    float *h_Y = (float*)malloc(M * sizeof(float));
    float *h_Cost = (float*)malloc(N * M * sizeof(float));

    for (int i = 0; i < N; i++) h_X[i] = (float)(rand() % 100);
    for (int j = 0; j < M; j++) h_Y[j] = (float)(rand() % 100);

    float *d_X, *d_Y, *d_Cost;
    cudaMalloc((void**)&d_X, N * sizeof(float));
    cudaMalloc((void**)&d_Y, M * sizeof(float));
    cudaMalloc((void**)&d_Cost, N * M * sizeof(float));

    cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeDTWCostMatrix<<<gridSize, blockSize>>>(d_X, d_Y, d_Cost, N, M);

    cudaMemcpy(h_Cost, d_Cost, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    printf("DTW Distance: %f\n", h_Cost[(N - 1) * M + (M - 1)]);

    free(h_X); free(h_Y); free(h_Cost);
    cudaFree(d_X); cudaFree(d_Y); cudaFree(d_Cost);

    return 0;
}
