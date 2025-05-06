#include <stdio.h>
#include <cuda_runtime.h>

// Hinge loss: L = max(0, 1 - y * y_pred)
__global__ void hinge_loss_kernel(const float* y_pred, const float* y_true, float* losses, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float margin = 1.0f - y_true[idx] * y_pred[idx];
    losses[idx] = margin > 0.0f ? margin : 0.0f;
}

int main() {
    const int size = 8;
    float h_pred[size] = {0.8, -0.5, 0.2, -0.9, 1.2, -1.0, 0.0, 0.3};   // predicted values
    float h_true[size] = {1, -1, 1, -1, 1, -1, 1, -1};                 // true labels (+1 or -1)
    float h_loss[size];

    float *d_pred, *d_true, *d_loss;
    cudaMalloc(&d_pred, size * sizeof(float));
    cudaMalloc(&d_true, size * sizeof(float));
    cudaMalloc(&d_loss, size * sizeof(float));

    cudaMemcpy(d_pred, h_pred, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_true, h_true, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    hinge_loss_kernel<<<blocks, threads>>>(d_pred, d_true, d_loss, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_loss, d_loss, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Hinge Loss Results:\n");
    for (int i = 0; i < size; ++i) {
        printf("True: %.1f, Pred: %.2f, Loss: %.4f\n", h_true[i], h_pred[i], h_loss[i]);
    }

    cudaFree(d_pred);
    cudaFree(d_true);
    cudaFree(d_loss);

    return 0;
}
