#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void checkCublasErrors(cublasStatus_t err) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d\n", err);
        exit(EXIT_FAILURE);
    }
}

int main() {
    cublasHandle_t handle;
    checkCublasErrors(cublasCreate(&handle));

    int M = 3, N = 2, K = 4;
    float *h_A, *h_B, *h_C;

    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices with different values
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = (float)(i * K + j);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = (float)(i * N + j + 1);

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, M * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f, beta = 0.0f;
    checkCublasErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  M, N, K, &alpha,
                                  d_A, M, d_B, K,
                                  &beta, d_C, M));

    checkCudaErrors(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    printf("Matrix C = A * B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    checkCublasErrors(cublasDestroy(handle));

    return 0;
}
