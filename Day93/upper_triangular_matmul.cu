#include <stdio.h>
#include <cuda_runtime.h>

__global__ void upperTriangularMatMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    // Optionally zero out lower triangle
    if (row < N && col < N && row > col) {
        C[row * N + col] = 0.0f;
    }
}

void printMatrix(const float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%5.1f ", mat[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    const int N = 4;
    const size_t size = N * N * sizeof(float);

    float h_A[N * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 1, 2, 3,
        4, 5, 6, 7
    };
    float h_B[N * N] = {
        7, 6, 5, 4,
        3, 2, 1, 0,
        1, 2, 3, 4,
        5, 6, 7, 8
    };
    float h_C[N * N];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    upperTriangularMatMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix (Upper Triangular MatMul):\n");
    printMatrix(h_C, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
