#include <stdio.h>
#include <cuda.h>

#define NUM_TOKENS 4
#define VOCAB_SIZE 10
#define EMBEDDING_DIM 8

// CUDA Kernel for embedding lookup
__global__ void embeddingLookupKernel(int* tokenIds, float* embeddingMatrix, float* outputEmbeddings) {
    int tokenIdx = blockIdx.x;   // each block handles one token
    int dimIdx = threadIdx.x;    // each thread handles one dimension

    if (dimIdx < EMBEDDING_DIM && tokenIdx < NUM_TOKENS) {
        int tokenId = tokenIds[tokenIdx];
        outputEmbeddings[tokenIdx * EMBEDDING_DIM + dimIdx] =
            embeddingMatrix[tokenId * EMBEDDING_DIM + dimIdx];
    }
}

int main() {
    // Input tokens
    int h_tokenIds[NUM_TOKENS] = {2, 5, 7, 1};

    // Random embedding matrix (VOCAB_SIZE x EMBEDDING_DIM)
    float h_embeddingMatrix[VOCAB_SIZE * EMBEDDING_DIM];
    for (int i = 0; i < VOCAB_SIZE * EMBEDDING_DIM; i++) {
        h_embeddingMatrix[i] = (float)(i % 10);  // just dummy values
    }

    // Output embedding vectors
    float h_outputEmbeddings[NUM_TOKENS * EMBEDDING_DIM];

    // Device memory
    int* d_tokenIds;
    float *d_embeddingMatrix, *d_outputEmbeddings;

    cudaMalloc(&d_tokenIds, NUM_TOKENS * sizeof(int));
    cudaMalloc(&d_embeddingMatrix, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_outputEmbeddings, NUM_TOKENS * EMBEDDING_DIM * sizeof(float));

    cudaMemcpy(d_tokenIds, h_tokenIds, NUM_TOKENS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embeddingMatrix, h_embeddingMatrix, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel: one block per token, EMBEDDING_DIM threads per block
    embeddingLookupKernel<<<NUM_TOKENS, EMBEDDING_DIM>>>(d_tokenIds, d_embeddingMatrix, d_outputEmbeddings);

    cudaMemcpy(h_outputEmbeddings, d_outputEmbeddings, NUM_TOKENS * EMBEDDING_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // Display result
    printf("Embedding Vectors:\n");
    for (int i = 0; i < NUM_TOKENS; i++) {
        printf("Token %d: ", h_tokenIds[i]);
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            printf("%.1f ", h_outputEmbeddings[i * EMBEDDING_DIM + j]);
        }
        printf("\n");
    }

    cudaFree(d_tokenIds);
    cudaFree(d_embeddingMatrix);
    cudaFree(d_outputEmbeddings);

    return 0;
}
