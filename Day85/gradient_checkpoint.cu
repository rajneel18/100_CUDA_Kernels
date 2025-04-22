#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 1024

__global__ void forward_kernel(float* input, float* output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        output[idx] = input[idx] * input[idx]; // Example: f(x) = x^2
}

__global__ void recompute_kernel(float* input, float* checkpoint, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        checkpoint[idx] = input[idx] * input[idx]; // Recompute for backward
}

__global__ void backward_kernel(float* input, float* grad_output, float* grad_input, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        grad_input[idx] = 2.0f * input[idx] * grad_output[idx]; // d(f)/dx = 2x
}

int main() {
    float *h_input, *h_grad_output, *h_grad_input;
    float *d_input, *d_output, *d_checkpoint, *d_grad_output, *d_grad_input;

    h_input = (float*)malloc(SIZE * sizeof(float));
    h_grad_output = (float*)malloc(SIZE * sizeof(float));
    h_grad_input = (float*)malloc(SIZE * sizeof(float));

    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 0.001f * i;
        h_grad_output[i] = 1.0f;
    }

    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));
    cudaMalloc(&d_checkpoint, SIZE * sizeof(float));
    cudaMalloc(&d_grad_output, SIZE * sizeof(float));
    cudaMalloc(&d_grad_input, SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // --- Forward Pass ---
    forward_kernel<<<(SIZE + 255) / 256, 256>>>(d_input, d_output, SIZE);

    // Checkpointing: Do not store intermediate output

    // --- Backward Pass with Recomputation ---
    recompute_kernel<<<(SIZE + 255) / 256, 256>>>(d_input, d_checkpoint, SIZE);
    backward_kernel<<<(SIZE + 255) / 256, 256>>>(d_input, d_grad_output, d_grad_input, SIZE);

    cudaMemcpy(h_grad_input, d_grad_input, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Gradients (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", h_grad_input[i]);
    }
    printf("\n");

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_checkpoint);
    cudaFree(d_grad_output); cudaFree(d_grad_input);
    free(h_input); free(h_grad_output); free(h_grad_input);

    return 0;
}
