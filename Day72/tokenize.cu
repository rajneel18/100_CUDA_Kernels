#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_TOKENS_PER_SEQ 8
#define MAX_WORD_LEN 16
#define NUM_SEQS 4
#define VOCAB_SIZE 6

// Simulated vocabulary (stored in constant memory)
__constant__ char d_vocab_words[VOCAB_SIZE][MAX_WORD_LEN] = {
    "hello", "world", "gpu", "cuda", "rocks", "ai"
};
__constant__ int d_vocab_ids[VOCAB_SIZE] = {
    1, 2, 3, 4, 5, 6
};

// GPU Kernel to convert tokens to IDs
__global__ void tokenize_kernel(char* token_matrix, int* token_ids, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        char* token = token_matrix + idx * MAX_WORD_LEN;

        for (int i = 0; i < VOCAB_SIZE; ++i) {
            bool match = true;
            for (int j = 0; j < MAX_WORD_LEN; ++j) {
                if (token[j] != d_vocab_words[i][j]) {
                    match = false;
                    break;
                }
                if (token[j] == '\0') break;
            }

            if (match) {
                token_ids[idx] = d_vocab_ids[i];
                return;
            }
        }
        // Unknown token
        token_ids[idx] = 0;
    }
}

// Host-side helper to split strings
void split_sequence(const char* seq, char* out, int seq_idx) {
    int token_count = 0;
    char buffer[256];
    strcpy(buffer, seq);
    char* token = strtok(buffer, " ");

    while (token && token_count < MAX_TOKENS_PER_SEQ) {
        strncpy(&out[(seq_idx * MAX_TOKENS_PER_SEQ + token_count) * MAX_WORD_LEN], token, MAX_WORD_LEN);
        token_count++;
        token = strtok(NULL, " ");
    }
}

int main() {
    const char* sequences[NUM_SEQS] = {
        "hello world cuda",
        "gpu rocks ai",
        "hello gpu",
        "cuda rocks world"
    };

    int total_tokens = NUM_SEQS * MAX_TOKENS_PER_SEQ;

    char* h_token_matrix = (char*)calloc(total_tokens * MAX_WORD_LEN, sizeof(char));
    int* h_token_ids = (int*)calloc(total_tokens, sizeof(int));

    // Tokenize on CPU
    for (int i = 0; i < NUM_SEQS; ++i)
        split_sequence(sequences[i], h_token_matrix, i);

    // Allocate GPU memory
    char* d_token_matrix;
    int* d_token_ids;
    cudaMalloc(&d_token_matrix, total_tokens * MAX_WORD_LEN * sizeof(char));
    cudaMalloc(&d_token_ids, total_tokens * sizeof(int));

    // Copy tokens to GPU
    cudaMemcpy(d_token_matrix, h_token_matrix, total_tokens * MAX_WORD_LEN * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 128;
    int blocks = (total_tokens + threads - 1) / threads;
    tokenize_kernel<<<blocks, threads>>>(d_token_matrix, d_token_ids, total_tokens);

    // Copy results back
    cudaMemcpy(h_token_ids, d_token_ids, total_tokens * sizeof(int), cudaMemcpyDeviceToHost);

    // Display result
    for (int i = 0; i < NUM_SEQS; ++i) {
        printf("Sequence %d token IDs: ", i);
        for (int j = 0; j < MAX_TOKENS_PER_SEQ; ++j) {
            int tid = h_token_ids[i * MAX_TOKENS_PER_SEQ + j];
            if (tid != 0) printf("%d ", tid);
        }
        printf("\n");
    }

    cudaFree(d_token_matrix);
    cudaFree(d_token_ids);
    free(h_token_matrix);
    free(h_token_ids);
    return 0;
}
