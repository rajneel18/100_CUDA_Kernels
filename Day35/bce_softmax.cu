#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmaxCrossEntropyFusedKernel(float *y_true, float *X, float *W, float *loss, int N, int D) {
    extern __shared__ float sharedMem[];
    float *logits = sharedMem;

    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx >= N) return;

    int true_label = (int)y_true[sample_idx];

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        logits[d] = 0.0f;
        for (int j = 0; j < D; j++) {
            logits[d] += X[sample_idx * D + j] * W[j * D + d];
        }
    }
    __syncthreads();

    float max_logit = -1e20f;
    for (int d = 0; d < D; d++) {
        max_logit = fmaxf(max_logit, logits[d]);
    }

    float sum_exp = 0.0f;
    for (int d = 0; d < D; d++) {
        logits[d] = expf(fmaxf(logits[d] - max_logit, -50.0f));
        sum_exp += logits[d];
    }

    for (int d = 0; d < D; d++) {
        logits[d] /= sum_exp;
    }

    float sample_loss = -logf(fmaxf(logits[true_label], 1e-7));

    __shared__ float block_loss;
    if (threadIdx.x == 0) block_loss = 0.0f;
    __syncthreads();

    atomicAdd(&block_loss, sample_loss);
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(loss, block_loss);
}

float softmaxCrossEntropyCUDA(float *y_true, float *X, float *W, int N, int D) {
    float *d_y_true, *d_X, *d_W, *d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_y_true, N * sizeof(float));
    cudaMalloc(&d_X, N * D * sizeof(float));
    cudaMalloc(&d_W, D * D * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_y_true, y_true, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, D * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = D * sizeof(float);

    softmaxCrossEntropyFusedKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_y_true, d_X, d_W, d_loss, N, D);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y_true);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_loss);

    return h_loss / N;
}

int main() {
    int N = 1000;
    int D = 10;

    float *y_true = (float*)malloc(N * sizeof(float));
    float *X = (float*)malloc(N * D * sizeof(float));
    float *W = (float*)malloc(D * D * sizeof(float));

    for (int i = 0; i < N; i++) {
        y_true[i] = rand() % D;
        for (int d = 0; d < D; d++) {
            X[i * D + d] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int d = 0; d < D * D; d++) {
        W[d] = static_cast<float>(rand()) / RAND_MAX;
    }

    float loss = softmaxCrossEntropyCUDA(y_true, X, W, N, D);

    printf("Optimized Softmax Cross-Entropy Loss: %f \n", loss);

    free(y_true);
    free(X);
    free(W);

    return 0;
}
