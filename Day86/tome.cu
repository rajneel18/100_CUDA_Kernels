#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NUM_TOKENS 8
#define EMBED_DIM 4
#define MERGE_PAIRS 2  // Number of pairs to merge

__device__ float cosine_similarity(float* a, float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot / (sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f);
}

__global__ void merge_tokens(float* tokens, int num_tokens, int embed_dim) {
    int idx = threadIdx.x;

    if (idx < MERGE_PAIRS) {
        int i = idx * 2;
        int j = i + 1;

        float* token_i = &tokens[i * embed_dim];
        float* token_j = &tokens[j * embed_dim];

        // Merge: simple average
        for (int k = 0; k < embed_dim; ++k) {
            float avg = 0.5f * (token_i[k] + token_j[k]);
            token_i[k] = avg;
            token_j[k] = avg;
        }
    }
}

int main() {
    float h_tokens[NUM_TOKENS * EMBED_DIM] = {
        0.1, 0.2, 0.3, 0.4,
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.4, 0.3, 0.2,
        0.5, 0.4, 0.3, 0.2,
        0.9, 0.8, 0.7, 0.6,
        0.91, 0.81, 0.71, 0.61,
        0.15, 0.25, 0.35, 0.45,
        0.16, 0.26, 0.36, 0.46
    };

    float* d_tokens;
    cudaMalloc(&d_tokens, sizeof(h_tokens));
    cudaMemcpy(d_tokens, h_tokens, sizeof(h_tokens), cudaMemcpyHostToDevice);

    merge_tokens<<<1, MERGE_PAIRS>>>(d_tokens, NUM_TOKENS, EMBED_DIM);

    cudaMemcpy(h_tokens, d_tokens, sizeof(h_tokens), cudaMemcpyDeviceToHost);

    printf("Merged Tokens:\n");
    for (int i = 0; i < NUM_TOKENS; i++) {
        printf("Token %d: ", i);
        for (int j = 0; j < EMBED_DIM; j++) {
            printf("%.2f ", h_tokens[i * EMBED_DIM + j]);
        }
        printf("\n");
    }

    cudaFree(d_tokens);
    return 0;
}

