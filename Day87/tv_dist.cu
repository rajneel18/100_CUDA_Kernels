// tv_distance.cu - CUDA C implementation of Total Variation Distance

#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void tv_distance_kernel(float* P, float* Q, float* result, int N) {
    extern __shared__ float temp[];

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    float diff = 0.0f;
    if (idx < N) {
        diff = fabsf(P[idx] - Q[idx]) * 0.5f;
    }

    temp[tid] = (idx < N) ? diff : 0.0f;
    __syncthreads();

    // Reduction to sum up the partial TV distances
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // First thread in block writes result
    if (tid == 0) {
        atomicAdd(result, temp[0]);
    }
}

int main() {
    const int N = 1024;
    float h_P[N], h_Q[N], h_result = 0.0f;
    for (int i = 0; i < N; ++i) {
        h_P[i] = 1.0f / N;
        h_Q[i] = (i % 2 == 0) ? 2.0f / N : 0.0f;
    }

    float *d_P, *d_Q, *d_result;
    cudaMalloc((void**)&d_P, N * sizeof(float));
    cudaMalloc((void**)&d_Q, N * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_P, h_P, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tv_distance_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_P, d_Q, d_result, N);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Total Variation Distance: %f\n", h_result);

    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_result);
    return 0;
}
