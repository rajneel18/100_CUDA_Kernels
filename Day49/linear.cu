#include <stdio.h>
#include <cuda_runtime.h>

#define N 3  // Matrix size

__global__ void forward_substitution(float *L, float *b, float *y) {
    int i = threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < i; j++) {
            sum += L[i * N + j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i * N + i];
    }
}

__global__ void backward_substitution(float *U, float *y, float *x) {
    int i = N - 1 - threadIdx.x;
    if (i >= 0) {
        float sum = 0.0f;
        for (int j = i + 1; j < N; j++) {
            sum += U[i * N + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * N + i];
    }
}

__global__ void lu_decomposition(float *A, float *L, float *U) {
    int i = threadIdx.x;
    if (i < N) {
        for (int j = i; j < N; j++) {
            U[i * N + j] = A[i * N + j];
            for (int k = 0; k < i; k++) {
                U[i * N + j] -= L[i * N + k] * U[k * N + j];
            }
        }
        for (int j = i; j < N; j++) {
            if (i == j)
                L[i * N + i] = 1.0f;
            else {
                L[j * N + i] = A[j * N + i];
                for (int k = 0; k < i; k++) {
                    L[j * N + i] -= L[j * N + k] * U[k * N + i];
                }
                L[j * N + i] /= U[i * N + i];
            }
        }
    }
}

int main() {
    float h_A[N * N] = {1.0f, 2.0f, 3.0f, 
                        4.0f, 5.0f, 6.0f, 
                        7.0f, 8.0f, 10.0f};
    float h_B[N] = {1.0f, 2.0f, 3.0f};
    float h_L[N * N] = {0}, h_U[N * N] = {0};
    float h_Y[N] = {0}, h_X[N] = {0};

    float *d_A, *d_L, *d_U, *d_B, *d_Y, *d_X;

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_L, N * N * sizeof(float));
    cudaMalloc((void**)&d_U, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_Y, N * sizeof(float));
    cudaMalloc((void**)&d_X, N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform LU decomposition
    lu_decomposition<<<1, N>>>(d_A, d_L, d_U);
    cudaDeviceSynchronize();

    // Solve Ly = B using forward substitution
    forward_substitution<<<1, N>>>(d_L, d_B, d_Y);
    cudaDeviceSynchronize();

    // Solve Ux = y using backward substitution
    backward_substitution<<<1, N>>>(d_U, d_Y, d_X);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_X, d_X, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print solution
    printf("Solution X:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_X[i]);
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_B);
    cudaFree(d_Y);
    cudaFree(d_X);

    return 0;
}
