#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 64  // Query length
#define M 64  // Key/Value length
#define D 32  // Embedding dimension
#define SCALE (1.0f / sqrtf(D))

__global__ void flash_attention_backward(
    float* Q, float* K, float* V,
    float* dO, float* softmax_out,
    float* dQ, float* dK, float* dV
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query index
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // key index

    if (i < N && j < M) {
        float grad_output_dot_v = 0.0f;
        for (int d = 0; d < D; ++d) {
            grad_output_dot_v += dO[i * D + d] * V[j * D + d];
        }

        float grad_softmax = grad_output_dot_v * softmax_out[i * M + j];

        // Compute dQ
        for (int d = 0; d < D; ++d) {
            atomicAdd(&dQ[i * D + d], grad_softmax * K[j * D + d] * SCALE);
        }

        // Compute dK
        for (int d = 0; d < D; ++d) {
            atomicAdd(&dK[j * D + d], grad_softmax * Q[i * D + d] * SCALE);
        }

        // Compute dV
        for (int d = 0; d < D; ++d) {
            atomicAdd(&dV[j * D + d], softmax_out[i * M + j] * dO[i * D + d]);
        }
    }
}

int main() {
    // Host memory allocation
    float *h_Q = (float*)malloc(N * D * sizeof(float));
    float *h_K = (float*)malloc(M * D * sizeof(float));
    float *h_V = (float*)malloc(M * D * sizeof(float));
    float *h_dO = (float*)malloc(N * D * sizeof(float));
    float *h_softmax_out = (float*)malloc(N * M * sizeof(float));
    float *h_dQ = (float*)malloc(N * D * sizeof(float));
    float *h_dK = (float*)malloc(M * D * sizeof(float));
    float *h_dV = (float*)malloc(M * D * sizeof(float));

    // Initialize host memory (e.g., with random values)
    for (int i = 0; i < N * D; ++i) h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * D; ++i) h_K[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < M * D; ++i) h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * D; ++i) h_dO[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * M; ++i) h_softmax_out[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory allocation
    float *d_Q, *d_K, *d_V, *d_dO, *d_softmax_out, *d_dQ, *d_dK, *d_dV;
    cudaMalloc(&d_Q, N * D * sizeof(float));
    cudaMalloc(&d_K, M * D * sizeof(float));
    cudaMalloc(&d_V, M * D * sizeof(float));
    cudaMalloc(&d_dO, N * D * sizeof(float));
    cudaMalloc(&d_softmax_out, N * M * sizeof(float));
    cudaMalloc(&d_dQ, N * D * sizeof(float));
    cudaMalloc(&d_dK, M * D * sizeof(float));
    cudaMalloc(&d_dV, M * D * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, M * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, M * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_softmax_out, h_softmax_out, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threads(8, 8);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    flash_attention_backward<<<blocks, threads>>>(
        d_Q, d_K, d_V, d_dO, d_softmax_out, d_dQ, d_dK, d_dV
    );
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_dQ, d_dQ, N * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dK, d_dK, M * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dV, d_dV, M * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_dO);
    cudaFree(d_softmax_out);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);

    // Free host memory
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_dO);
    free(h_softmax_out);
    free(h_dQ);
    free(h_dK);
    free(h_dV);

    return 0;
}
