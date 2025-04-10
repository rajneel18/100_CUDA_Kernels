#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024     // Number of samples
#define D 32       // Number of features

// CUDA kernel to compute column means
__global__ void compute_means(float* data, float* means) {
    int col = threadIdx.x;
    if (col < D) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += data[i * D + col];
        }
        means[col] = sum / N;
    }
}

// CUDA kernel to center the data matrix (subtract mean)
__global__ void center_data(float* data, float* means) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
        int col = idx % D;
        data[idx] -= means[col];
    }
}

// CUDA kernel to compute covariance matrix (upper triangle only)
__global__ void compute_covariance(float* data, float* cov_matrix) {
    int i = threadIdx.y;
    int j = threadIdx.x;

    if (i < D && j < D && j >= i) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float xi = data[k * D + i];
            float xj = data[k * D + j];
            sum += xi * xj;
        }
        cov_matrix[i * D + j] = sum / (N - 1);
    }
}

int main() {
    size_t data_size = N * D * sizeof(float);
    size_t cov_size = D * D * sizeof(float);

    // Host allocations
    float *h_data = (float*)malloc(data_size);
    float *h_cov = (float*)malloc(cov_size);

    // Initialize data with random floats
    for (int i = 0; i < N * D; i++) {
        h_data[i] = ((float)rand() / RAND_MAX);
    }

    // Device allocations
    float *d_data, *d_means, *d_cov;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_means, D * sizeof(float));
    cudaMalloc(&d_cov, cov_size);

    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    // Step 1: Compute column-wise means
    compute_means<<<1, D>>>(d_data, d_means);

    // Step 2: Center the data (X - mean)
    center_data<<<(N * D + 255)/256, 256>>>(d_data, d_means);

    // Step 3: Compute Covariance Matrix
    dim3 threads(D, D); // D x D threads (only upper triangle needed)
    compute_covariance<<<1, threads>>>(d_data, d_cov);

    cudaMemcpy(h_cov, d_cov, cov_size, cudaMemcpyDeviceToHost);

    // Print partial covariance matrix
    printf("Covariance Matrix (upper triangle):\n");
    for (int i = 0; i < D; i++) {
        for (int j = 0; j < D; j++) {
            if (j >= i) printf("%6.3f ", h_cov[i * D + j]);
            else printf("   .   "); // skip lower triangle
        }
        printf("\n");
    }

    cudaFree(d_data); cudaFree(d_means); cudaFree(d_cov);
    free(h_data); free(h_cov);
    return 0;
}
