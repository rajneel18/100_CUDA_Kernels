#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ float dot_product(const float* a, const float* b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}

__device__ float norm(const float* v, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i)
        sum += v[i] * v[i];
    return sqrtf(sum);
}

__global__ void negative_cosine_similarity_kernel(
    const float* x1, const float* x2, float* output,
    int batch_size, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* v1 = x1 + idx * dim;
    const float* v2 = x2 + idx * dim;

    float dp = dot_product(v1, v2, dim);
    float n1 = norm(v1, dim);
    float n2 = norm(v2, dim);

    float cos_sim = dp / (n1 * n2 + 1e-8f);
    output[idx] = -cos_sim;
}

int main() {
    const int batch_size = 4;
    const int dim = 3;

    float h_x1[batch_size * dim] = {
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
        0, 0, 1
    };

    float h_x2[batch_size * dim] = {
        1, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0
    };

    float h_output[batch_size];

    float *d_x1, *d_x2, *d_output;
    cudaMalloc(&d_x1, batch_size * dim * sizeof(float));
    cudaMalloc(&d_x2, batch_size * dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * sizeof(float));

    cudaMemcpy(d_x1, h_x1, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    negative_cosine_similarity_kernel<<<blocks, threads>>>(d_x1, d_x2, d_output, batch_size, dim);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Negative Cosine Similarity:\n");
    for (int i = 0; i < batch_size; ++i) {
        printf("Sample %d: %.6f\n", i, h_output[i]);
    }

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_output);

    return 0;
}
