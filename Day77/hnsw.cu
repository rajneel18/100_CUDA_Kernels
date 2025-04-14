// hnsw_knn.cu - Approximate Nearest Neighbor Search (simplified HNSW-style) in CUDA

#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define NUM_VECTORS 1024
#define VECTOR_DIM 64
#define K_NEIGHBORS 5

__device__ float l2_distance(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

__global__ void knn_search(const float* __restrict__ dataset, const float* __restrict__ queries, int* __restrict__ neighbors) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;  // Query ID
    if (qid >= NUM_VECTORS) return;

    const float* query = &queries[qid * VECTOR_DIM];

    float best_dists[K_NEIGHBORS];
    int best_indices[K_NEIGHBORS];
    for (int i = 0; i < K_NEIGHBORS; ++i) {
        best_dists[i] = 1e30;
        best_indices[i] = -1;
    }

    for (int vid = 0; vid < NUM_VECTORS; ++vid) {
        const float* vec = &dataset[vid * VECTOR_DIM];
        float dist = l2_distance(query, vec, VECTOR_DIM);

        // Find position to insert
        for (int k = 0; k < K_NEIGHBORS; ++k) {
            if (dist < best_dists[k]) {
                // Shift others
                for (int s = K_NEIGHBORS - 1; s > k; --s) {
                    best_dists[s] = best_dists[s - 1];
                    best_indices[s] = best_indices[s - 1];
                }
                best_dists[k] = dist;
                best_indices[k] = vid;
                break;
            }
        }
    }

    for (int i = 0; i < K_NEIGHBORS; ++i) {
        neighbors[qid * K_NEIGHBORS + i] = best_indices[i];
    }
}

int main() {
    float *h_dataset = new float[NUM_VECTORS * VECTOR_DIM];
    float *h_queries = new float[NUM_VECTORS * VECTOR_DIM];
    int *h_neighbors = new int[NUM_VECTORS * K_NEIGHBORS];

    // Initialize with random data
    for (int i = 0; i < NUM_VECTORS * VECTOR_DIM; ++i) {
        h_dataset[i] = rand() / (float)RAND_MAX;
        h_queries[i] = rand() / (float)RAND_MAX;
    }

    float *d_dataset, *d_queries;
    int *d_neighbors;

    cudaMalloc(&d_dataset, NUM_VECTORS * VECTOR_DIM * sizeof(float));
    cudaMalloc(&d_queries, NUM_VECTORS * VECTOR_DIM * sizeof(float));
    cudaMalloc(&d_neighbors, NUM_VECTORS * K_NEIGHBORS * sizeof(int));

    cudaMemcpy(d_dataset, h_dataset, NUM_VECTORS * VECTOR_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, NUM_VECTORS * VECTOR_DIM * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(128);
    dim3 gridSize((NUM_VECTORS + blockSize.x - 1) / blockSize.x);
    knn_search<<<gridSize, blockSize>>>(d_dataset, d_queries, d_neighbors);

    cudaMemcpy(h_neighbors, d_neighbors, NUM_VECTORS * K_NEIGHBORS * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; ++i) {
        printf("Query %d Neighbors: ", i);
        for (int j = 0; j < K_NEIGHBORS; ++j) {
            printf("%d ", h_neighbors[i * K_NEIGHBORS + j]);
        }
        printf("\n");
    }

    cudaFree(d_dataset);
    cudaFree(d_queries);
    cudaFree(d_neighbors);
    delete[] h_dataset;
    delete[] h_queries;
    delete[] h_neighbors;

    return 0;
}
