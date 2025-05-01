#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void huber_loss_kernel(const float* y_true, const float* y_pred, float* loss, int N, float delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float diff = y_true[i] - y_pred[i];
    float abs_diff = fabsf(diff);

    if (abs_diff <= delta) {
        loss[i] = 0.5f * diff * diff;
    } else {
        loss[i] = delta * (abs_diff - 0.5f * delta);
    }
}

void print_array(const float* arr, int N) {
    for (int i = 0; i < N; ++i) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

int main() {
    const int N = 8;
    const float delta = 1.0f;

    float h_y_true[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_y_pred[N] = {1, 2.5, 2.8, 3.2, 5.5, 6, 8.5, 7};
    float h_loss[N];

    float *d_y_true, *d_y_pred, *d_loss;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_y_true, size);
    cudaMalloc(&d_y_pred, size);
    cudaMalloc(&d_loss, size);

    cudaMemcpy(d_y_true, h_y_true, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, h_y_pred, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    huber_loss_kernel<<<blocks, threads>>>(d_y_true, d_y_pred, d_loss, N, delta);
    cudaDeviceSynchronize();

    cudaMemcpy(h_loss, d_loss, size, cudaMemcpyDeviceToHost);

    printf("Huber Loss Output:\n");
    print_array(h_loss, N);

    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_loss);

    return 0;
}
