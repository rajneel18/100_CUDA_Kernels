#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N 256    // Number of document vectors
#define D 128    // Dimensions
#define K 5      // Top-K documents

// CUDA kernel for cosine similarity
__global__ void compute_cosine_similarity(float* docs, float* query, float* scores, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float dot = 0.0f;
    float norm_doc = 0.0f;
    float norm_query = 0.0f;

    for (int i = 0; i < d; ++i) {
        float doc_val = docs[idx * d + i];
        float query_val = query[i];
        dot += doc_val * query_val;
        norm_doc += doc_val * doc_val;
        norm_query += query_val * query_val;
    }

    norm_doc = sqrtf(norm_doc);
    norm_query = sqrtf(norm_query);

    if (norm_doc > 1e-6f && norm_query > 1e-6f) {
        scores[idx] = dot / (norm_doc * norm_query);
    } else {
        scores[idx] = 0.0f;
    }
}

// Utility to find top K indices with highest scores
void find_top_k(float* scores, int* indices, int total, int k) {
    for (int i = 0; i < k; ++i) {
        float max_val = -1.0f;
        int max_idx = -1;
        for (int j = 0; j < total; ++j) {
            if (scores[j] > max_val) {
                max_val = scores[j];
                max_idx = j;
            }
        }
        indices[i] = max_idx;
        scores[max_idx] = -9999.0f;  // Exclude in next round
    }
}

int main() {
    float *h_docs, *h_query, *h_scores;
    float *d_docs, *d_query, *d_scores;

    h_docs = (float*)malloc(N * D * sizeof(float));
    h_query = (float*)malloc(D * sizeof(float));
    h_scores = (float*)malloc(N * sizeof(float));

    // Initialize random document embeddings and query
    for (int i = 0; i < N * D; ++i)
        h_docs[i] = ((float)rand() / RAND_MAX);
    for (int i = 0; i < D; ++i)
        h_query[i] = ((float)rand() / RAND_MAX);

    cudaMalloc(&d_docs, N * D * sizeof(float));
    cudaMalloc(&d_query, D * sizeof(float));
    cudaMalloc(&d_scores, N * sizeof(float));

    cudaMemcpy(d_docs, h_docs, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, h_query, D * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    compute_cosine_similarity<<<blocks, threads>>>(d_docs, d_query, d_scores, N, D);
    cudaDeviceSynchronize();

    cudaMemcpy(h_scores, d_scores, N * sizeof(float), cudaMemcpyDeviceToHost);

    int top_indices[K];
    find_top_k(h_scores, top_indices, N, K);

    printf("🔍 Top-%d Similar Documents (Cosine):\n", K);
    for (int i = 0; i < K; ++i) {
        printf("Doc %d → Similarity = %.4f\n", top_indices[i], h_scores[top_indices[i]]);
    }

    // Cleanup
    free(h_docs); free(h_query); free(h_scores);
    cudaFree(d_docs); cudaFree(d_query); cudaFree(d_scores);
    return 0;
}
