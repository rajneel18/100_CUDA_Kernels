// sparse_attention.cu
// CUDA C Implementation: Simplified Sparse Attention (Block-wise Masked Attention)

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define SEQ_LEN 16
#define BLOCK_SIZE 4
#define EMBED_DIM 8

__device__ float softmax(float* scores, int len, int idx) {
    float max_val = -1e9;
    for (int i = 0; i < len; i++) max_val = fmaxf(max_val, scores[i]);

    float sum = 0.0f;
    for (int i = 0; i < len; i++) sum += expf(scores[i] - max_val);

    return expf(scores[idx] - max_val) / sum;
}

__global__ void sparse_attention(float* Q, float* K, float* V, float* output) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int row = block_id * BLOCK_SIZE + thread_id;
    if (row >= SEQ_LEN) return;

    float scores[BLOCK_SIZE];
    float attn[BLOCK_SIZE];

    // Compute attention scores (only within block)
    for (int j = 0; j < BLOCK_SIZE; j++) {
        int col = block_id * BLOCK_SIZE + j;
        if (col >= SEQ_LEN) {
            scores[j] = -1e9;  // Masked
            continue;
        }
        float score = 0.0f;
        for (int k = 0; k < EMBED_DIM; k++) {
            score += Q[row * EMBED_DIM + k] * K[col * EMBED_DIM + k];
        }
        scores[j] = score / sqrtf(EMBED_DIM);
    }

    for (int j = 0; j < BLOCK_SIZE; j++) {
        attn[j] = softmax(scores, BLOCK_SIZE, j);
    }

    // Weighted sum
    for (int k = 0; k < EMBED_DIM; k++) {
        float val = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int col = block_id * BLOCK_SIZE + j;
            if (col >= SEQ_LEN) continue;
            val += attn[j] * V[col * EMBED_DIM + k];
        }
        output[row * EMBED_DIM + k] = val;
    }
}

void print_matrix(float* mat, int rows, int cols, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int size = SEQ_LEN * EMBED_DIM * sizeof(float);
    float *h_Q = (float*)malloc(size);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    // Initialize Q, K, V with sample values
    for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
        h_Q[i] = h_K[i] = h_V[i] = (i % 10 + 1) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, size);
    cudaMalloc(&d_K, size);
    cudaMalloc(&d_V, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    int num_blocks = (SEQ_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sparse_attention<<<num_blocks, BLOCK_SIZE>>>(d_Q, d_K, d_V, d_out);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    print_matrix(h_out, SEQ_LEN, EMBED_DIM, "Sparse Attention Output");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    free(h_Q); free(h_K); free(h_V); free(h_out);

    return 0;
}
