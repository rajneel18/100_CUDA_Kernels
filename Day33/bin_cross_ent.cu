#include <stdio.h>
#include <cuda_runtime.h>

__global__ void binaryCrossEntropyKernel(float *y_true, float *y_pred, float *loss, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float y = y_true[idx];
        float y_hat = y_pred[idx];

        // Avoid log(0) by restricting predictions
        y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7), 1e-7);

        // Compute BCE loss for 
        float sample_loss = -(y * logf(y_hat) + (1 - y) * logf(1 - y_hat));

        // Use atomicAdd to accumulate loss 
        atomicAdd(loss, sample_loss);
    }
}

float binaryCrossEntropyCUDA(float *y_true, float *y_pred, int N) {
    float *d_y_true, *d_y_pred, *d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_y_true, N * sizeof(float));
    cudaMalloc(&d_y_pred, N * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_y_true, y_true, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, y_pred, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    binaryCrossEntropyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y_true, d_y_pred, d_loss, N);

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_loss);

    return h_loss / N;
}

int main() {
    int N = 10;
    float y_true[10] = {1, 0, 1, 1, 0, 1, 0, 1, 0, 1};
    float y_pred[10] = {0.9, 0.1, 0.8, 0.7, 0.2, 0.85, 0.05, 0.95, 0.15, 0.98};

    float loss = binaryCrossEntropyCUDA(y_true, y_pred, N);
    printf("Binary Cross Entropy Loss: %f\n", loss);

    return 0;
}
