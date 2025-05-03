#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define IDX2C(i, j, ld) (((j)*(ld))+(i)) // Column-major index

__device__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void matmul_swish_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = swish(acc);
    }
}

void print_matrix(const float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.5f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    const int M = 2, K = 3, N = 2;
    float h_A[M*K] = {1, 2, 3,
                      4, 5, 6};
    float h_B[K*N] = {1, 4,
                      2, 5,
                      3, 6};
    float h_C[M*N];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (M + 15)/16);
    matmul_swish_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output (matmul + swish):\n");
    print_matrix(h_C, M, N);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
