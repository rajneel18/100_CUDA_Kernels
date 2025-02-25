#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Approximate GELU using tanh
__device__ float gelu(float x) {
    const float sqrt_2_over_pi = sqrtf(2.0f / M_PI);
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
}

// CUDA kernel to apply GELU activation
__global__ void geluKernel(float* d_out, float* d_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = gelu(d_in[idx]);
    }
}

int main() {
    const int N = 10;
    float h_in[N] = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0}; 
    float h_out[N];

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    geluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("GELU outputs:\n");
    for (int i = 0; i < N; i++) {
        printf("GELU(%.2f) = %.6f\n", h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
