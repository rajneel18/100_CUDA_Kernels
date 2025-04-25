// mrope.cu - Rotary Positional Embeddings using memory-efficient implementation
// Author: OpenAI user

#include <stdio.h>
#include <math.h>

#define N 128         // number of tokens
#define D 64          // embedding dimension
#define PI 3.14159265359f

__device__ void apply_rope(float* q, int dim, float theta_base) {
    for (int i = 0; i < dim / 2; ++i) {
        float theta = powf(theta_base, (float)i / (dim / 2));
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float x = q[i];
        float y = q[i + dim / 2];

        q[i] = x * cos_theta - y * sin_theta;
        q[i + dim / 2] = x * sin_theta + y * cos_theta;
    }
}

__global__ void mrope_kernel(float* Q, float theta_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float* q = Q + idx * D;
        apply_rope(q, D, theta_base);
    }
}

int main() {
    float* h_Q = (float*)malloc(N * D * sizeof(float));
    float* d_Q;
    cudaMalloc((void**)&d_Q, N * D * sizeof(float));

    // Initialize dummy embeddings
    for (int i = 0; i < N * D; ++i)
        h_Q[i] = (float)(i % 100) / 100.0f;

    cudaMemcpy(d_Q, h_Q, N * D * sizeof(float), cudaMemcpyHostToDevice);

    float theta_base = 10000.0f;

    mrope_kernel<<<(N + 31) / 32, 32>>>(d_Q, theta_base);
    cudaMemcpy(h_Q, d_Q, N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first transformed token
    printf("First token after RoPE:\n");
    for (int i = 0; i < D; ++i) {
        printf("%.4f ", h_Q[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    cudaFree(d_Q);
    free(h_Q);
    return 0;
}
