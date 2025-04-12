// cosine_similarity.cu
#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define N 4   // Number of vectors
#define D 3   // Dimensions per vector

__device__ float dot_product(const float *a, const float *b, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i)
        result += a[i] * b[i];
    return result;
}

__device__ float norm(const float *a, int dim) {
    float result = 0.0f;
    for (int i = 0; i < dim; ++i)
        result += a[i] * a[i];
    return sqrtf(result);
}

__global__ void cosine_similarity(float* vectors, float* similarity_matrix, int dim) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i < N && j < N) {
        float *vec1 = &vectors[i * dim];
        float *vec2 = &vectors[j * dim];

        float dp = dot_product(vec1, vec2, dim);
        float norm1 = norm(vec1, dim);
        float norm2 = norm(vec2, dim);

        similarity_matrix[i * N + j] = dp / (norm1 * norm2 + 1e-8); // avoid divide by zero
    }
}

int main() {
    float h_vectors[N * D] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };

    float *d_vectors, *d_output;
    float h_output[N * N];

    cudaMalloc(&d_vectors, sizeof(float) * N * D);
    cudaMalloc(&d_output, sizeof(float) * N * N);

    cudaMemcpy(d_vectors, h_vectors, sizeof(float) * N * D, cudaMemcpyHostToDevice);

    cosine_similarity<<<N, N>>>(d_vectors, d_output, D);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    printf("Cosine Similarity Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.4f ", h_output[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_vectors);
    cudaFree(d_output);
    return 0;
}
