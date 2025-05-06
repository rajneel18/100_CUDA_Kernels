#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// ELU activation function: f(x) = x if x > 0 else alpha * (exp(x) - 1)
__global__ void elu_kernel(const float* input, float* output, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
}

int main() {
    const int size = 10;
    float h_input[size] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, -3.0, 3.0, -0.1};
    float h_output[size];
    float *d_input, *d_output;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    float alpha = 1.0f;
    elu_kernel<<<blocks, threads>>>(d_input, d_output, size, alpha);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("ELU Activation Results (alpha = %.1f):\n", alpha);
    for (int i = 0; i < size; ++i) {
        printf("Input: %.2f -> Output: %.5f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
