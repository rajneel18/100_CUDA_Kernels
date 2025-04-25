#include <stdio.h>
#include <math.h>

#define N 1024  // Example vector length
#define EPS 1e-8

__global__ void jsd_kernel(float* P, float* Q, float* result) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float temp = 0.0f;

    if (i < N) {
        float m = 0.5f * (P[i] + Q[i]);

        if (P[i] > EPS)
            temp += 0.5f * P[i] * logf(P[i] / m + EPS);

        if (Q[i] > EPS)
            temp += 0.5f * Q[i] * logf(Q[i] / m + EPS);
    }

    cache[tid] = temp;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, cache[0]);
}

int main() {
    float h_P[N], h_Q[N], h_result = 0.0f;
    float *d_P, *d_Q, *d_result;

    // Initialize dummy distributions
    for (int i = 0; i < N; i++) {
        h_P[i] = 1.0f / N;
        h_Q[i] = (i % 2 == 0) ? 2.0f / N : 0.5f / N;
    }

    cudaMalloc(&d_P, N * sizeof(float));
    cudaMalloc(&d_Q, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_P, h_P, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    jsd_kernel<<<(N + 255) / 256, 256>>>(d_P, d_Q, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Jensen-Shannon Divergence: %f\n", h_result);

    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_result);

    return 0;
}
