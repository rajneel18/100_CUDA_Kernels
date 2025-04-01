#include <stdio.h>
#include <cuda_runtime.h>

#define N 256   // Number of points in the distributions
#define EPS 1e-3 // Regularization for stability

// Kernel to compute the cost matrix (Euclidean distance)
__global__ void compute_cost_matrix(float* cost, float* x, float* y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size && j < size) {
        float dist = fabs(x[i] - y[j]);
        cost[i * size + j] = dist;
    }
}

// Kernel for Sinkhorn-Knopp Iteration (Entropy Regularization)
__global__ void sinkhorn_update(float* u, float* v, float* cost, float lambda, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float sum = 0.0;
        for (int j = 0; j < size; j++) {
            sum += expf(-lambda * (cost[i * size + j] + v[j]));
        }
        u[i] = logf(1.0 / sum);
    }
}

// Final Wasserstein Distance Computation (Transport Cost Summation)
__global__ void compute_wasserstein(float* transport_cost, float* u, float* v, float* cost, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        for (int j = 0; j < size; j++) {
            atomicAdd(transport_cost, expf(u[i] + v[j]) * cost[i * size + j]);
        }
    }
}

int main() {
    // Host & Device Memory Allocations
    float *h_x, *h_y, *h_cost, *h_transport_cost;
    float *d_x, *d_y, *d_cost, *d_transport_cost, *d_u, *d_v;

    int size = N * sizeof(float);
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);
    h_cost = (float*)malloc(N * N * sizeof(float));
    h_transport_cost = (float*)malloc(sizeof(float));

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_cost, N * N * sizeof(float));
    cudaMalloc(&d_transport_cost, sizeof(float));
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_v, size);

    // Initialize distributions (Random values between 0 and 1)
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)rand() / RAND_MAX;
        h_y[i] = (float)rand() / RAND_MAX;
    }

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    cudaMemset(d_transport_cost, 0, sizeof(float));

    // Compute Cost Matrix
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);
    compute_cost_matrix<<<gridSize, blockSize>>>(d_cost, d_x, d_y, N);
    cudaDeviceSynchronize();

    // Sinkhorn Iterations (Approximating Optimal Transport)
    dim3 blockSize1(256);
    dim3 gridSize1((N + 255) / 256);
    for (int iter = 0; iter < 50; iter++) {
        sinkhorn_update<<<gridSize1, blockSize1>>>(d_u, d_v, d_cost, 1.0 / EPS, N);
        sinkhorn_update<<<gridSize1, blockSize1>>>(d_v, d_u, d_cost, 1.0 / EPS, N);
    }

    // Compute Wasserstein Distance
    compute_wasserstein<<<gridSize1, blockSize1>>>(d_transport_cost, d_u, d_v, d_cost, N);
    cudaMemcpy(h_transport_cost, d_transport_cost, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Approximate Wasserstein Distance: %f\n", *h_transport_cost);

    // Cleanup
    free(h_x); free(h_y); free(h_cost); free(h_transport_cost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_cost); cudaFree(d_transport_cost);
    cudaFree(d_u); cudaFree(d_v);

    return 0;
}
