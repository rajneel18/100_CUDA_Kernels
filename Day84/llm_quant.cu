// gqa.cu - Grouped Query Attention (GQA) CUDA Kernel

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define D 64     // hidden dim per head
#define N 128    // sequence length
#define H 8      // total number of heads
#define G 2      // number of query groups
#define Hk (H / G) // heads per key-value group

__global__ void gqa_attention(float* Q, float* K, float* V, float* output) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int head = bid % H;
    int token = bid / H;

    int group = head / Hk;
    int kv_head_start = group * Hk;

    float scores[N] = {0};
    float max_score = -INFINITY;

    // Compute dot product of Q with all K in the same group
    for (int j = 0; j < N; ++j) {
        float score = 0;
        for (int d = 0; d < D; ++d) {
            float q_val = Q[(token * H + head) * D + d];
            float k_val = K[(j * H + kv_head_start + head % Hk) * D + d];
            score += q_val * k_val;
        }
        score /= sqrtf(D);
        scores[j] = score;
        max_score = fmaxf(max_score, score);
    }

    // Softmax normalization
    float sum_exp = 0;
    for (int j = 0; j < N; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }

    for (int j = 0; j < N; ++j) {
        scores[j] /= sum_exp;
    }

    // Compute output as weighted sum of V
    for (int d = 0; d < D; ++d) {
        float val = 0;
        for (int j = 0; j < N; ++j) {
            float v_val = V[(j * H + kv_head_start + head % Hk) * D + d];
            val += scores[j] * v_val;
        }
        output[(token * H + head) * D + d] = val;
    }
}

int main() {
    int size = N * H * D * sizeof(float);
    float *Q, *K, *V, *output;

    cudaMallocManaged(&Q, size);
    cudaMallocManaged(&K, size);
    cudaMallocManaged(&V, size);
    cudaMallocManaged(&output, size);

    // Initialize Q, K, V with dummy values
    for (int i = 0; i < N * H * D; ++i) {
        Q[i] = 0.01f * (i % 100);
        K[i] = 0.02f * (i % 100);
        V[i] = 0.03f * (i % 100);
    }

    gqa_attention<<<N * H, 1>>>(Q, K, V, output);
    cudaDeviceSynchronize();

    printf("Sample output values: \n");
    for (int i = 0; i < 5 * D; ++i) {
        printf("%.4f ", output[i]);
        if ((i + 1) % D == 0) printf("\n");
    }

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(output);
    return 0;
}
