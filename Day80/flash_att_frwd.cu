#include <stdio.h>
#include <math.h>

#define B 1          // Batch size
#define H 1          // Number of heads
#define N 64         // Sequence length
#define D 64         // Head dimension

__global__ void flash_attention_forward(float* Q, float* K, float* V, float* output) {
    __shared__ float Q_tile[D];
    __shared__ float score[N];

    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (tid < D) {
        Q_tile[tid] = Q[row * D + tid];
    }
    __syncthreads();

    if (tid < N) {
        float dot = 0.0;
        for (int i = 0; i < D; ++i) {
            dot += Q_tile[i] * K[tid * D + i];
        }
        score[tid] = dot / sqrtf((float)D);
    }
    __syncthreads();

    if (tid < N) {
        float max_score = -1e9;
        for (int i = 0; i < N; ++i) max_score = fmaxf(max_score, score[i]);

        float denom = 0.0;
        for (int i = 0; i < N; ++i) denom += __expf(score[i] - max_score);

        float result = 0.0;
        for (int i = 0; i < N; ++i) {
            float weight = __expf(score[i] - max_score) / denom;
            result += weight * V[i * D + tid];
        }
        output[row * D + tid] = result;
    }
}

int main() {
    float *Q, *K, *V, *output;
    cudaMallocManaged(&Q, N * D * sizeof(float));
    cudaMallocManaged(&K, N * D * sizeof(float));
    cudaMallocManaged(&V, N * D * sizeof(float));
    cudaMallocManaged(&output, N * D * sizeof(float));

    for (int i = 0; i < N * D; ++i) {
        Q[i] = K[i] = V[i] = static_cast<float>(i % 10);
    }

    flash_attention_forward<<<N, D>>>(Q, K, V, output);
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; ++i) printf("%.2f ", output[i]);
    printf("\n");

    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(output);
    return 0;
}