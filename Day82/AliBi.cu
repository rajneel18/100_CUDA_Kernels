#include <stdio.h>
#include <cuda_runtime.h>

#define SEQ_LEN 8
#define HEADS 4

__global__ void alibi_attention_kernel(float* attn_scores, float slope, int seq_len) {
    int head = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;

    int idx = head * seq_len * seq_len + row * seq_len + col;
    int distance = col - row;

    if (col >= row) {
        float bias = -distance * slope;
        attn_scores[idx] += bias;
    }
}

void print_matrix(float* mat, int heads, int seq_len) {
    for (int h = 0; h < heads; ++h) {
        printf("Head %d:\n", h);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                printf("%6.2f ", mat[h * seq_len * seq_len + i * seq_len + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main() {
    const int size = HEADS * SEQ_LEN * SEQ_LEN;
    float slope = 0.1f;

    float* h_attn_scores = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) h_attn_scores[i] = 1.0f;

    float* d_attn_scores;
    cudaMalloc(&d_attn_scores, size * sizeof(float));
    cudaMemcpy(d_attn_scores, h_attn_scores, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(HEADS);
    dim3 block(SEQ_LEN, SEQ_LEN);
    alibi_attention_kernel<<<grid, block>>>(d_attn_scores, slope, SEQ_LEN);
    cudaDeviceSynchronize();

    cudaMemcpy(h_attn_scores, d_attn_scores, size * sizeof(float), cudaMemcpyDeviceToHost);

    print_matrix(h_attn_scores, HEADS, SEQ_LEN);

    cudaFree(d_attn_scores);
    free(h_attn_scores);

    return 0;
}
