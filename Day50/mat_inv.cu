#include <stdio.h>
#include <cuda_runtime.h>

#define N 3  // Matrix size

__global__ void gauss_jordan_elimination(float *A, float *I) {
    int col = threadIdx.x;
    
    for (int i = 0; i < N; i++) {
        float diag_element = A[i * N + i];

        if (col < N) {
            A[i * N + col] /= diag_element; // Normalize row
            I[i * N + col] /= diag_element;
        }
        __syncthreads(); // Synchronize after row normalization

        if (col < N && col != i) {
            float factor = A[col * N + i];
            for (int j = 0; j < N; j++) {
                A[col * N + j] -= factor * A[i * N + j];
                I[col * N + j] -= factor * I[i * N + j];
            }
        }
        __syncthreads(); // Synchronize after elimination step
    }
}

int main() {
    float h_A[N * N] = { 4, 7, 2,
                         3, 6, 1,
                         2, 5, 3 };

    float h_I[N * N] = {1, 0, 0,
                        0, 1, 0,
                        0, 0, 1}; // Identity matrix

    float *d_A, *d_I;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_I, N * N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_I, h_I, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    gauss_jordan_elimination<<<1, N>>>(d_A, d_I);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_I, d_I, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Inverted Matrix
    printf("Inverse Matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_I[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_I);

    return 0;
}
