#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SEQ_LEN 64   // Sequence length (S)
#define DIM 64       // Embedding dimension (D)

__global__ void attention_score_kernel(float* Q, float* K, float* score) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Q[i]
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K[j]

    if (row < SEQ_LEN && col < SEQ_LEN) {
        float sum = 0.0f;
        for (int d = 0; d < DIM; d++) {
            float q_val = Q[row * DIM + d];
            float k_val = K[col * DIM + d];  // Transposed implicitly
            sum += q_val * k_val;
        }
        score[row * SEQ_LEN + col] = sum / sqrtf((float)DIM); // Scaled dot product
    }
}

void initialize(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = ((float)(rand() % 100)) / 100.0f;
    }
}

int main() {
    size_t vecSize = SEQ_LEN * DIM * sizeof(float);
    size_t scoreSize = SEQ_LEN * SEQ_LEN * sizeof(float);

    float *h_Q = (float*)malloc(vecSize);
    float *h_K = (float*)malloc(vecSize);
    float *h_score = (float*)malloc(scoreSize);

    initialize(h_Q, SEQ_LEN * DIM);
    initialize(h_K, SEQ_LEN * DIM);

    float *d_Q, *d_K, *d_score;
    cudaMalloc(&d_Q, vecSize);
    cudaMalloc(&d_K, vecSize);
    cudaMalloc(&d_score, scoreSize);

    cudaMemcpy(d_Q, h_Q, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, vecSize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((SEQ_LEN + threads.x - 1) / threads.x, (SEQ_LEN + threads.y - 1) / threads.y);

    attention_score_kernel<<<blocks, threads>>>(d_Q, d_K, d_score);
    cudaDeviceSynchronize();

    cudaMemcpy(h_score, d_score, scoreSize, cudaMemcpyDeviceToHost);

    printf("Attention score matrix (first 5x5 block):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", h_score[i * SEQ_LEN + j]);
        }
        printf("\n");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_score);
    free(h_Q); free(h_K); free(h_score);
    return 0;
}
