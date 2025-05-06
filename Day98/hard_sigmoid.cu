#include <stdio.h>
#include <cuda_runtime.h>

// Hard Sigmoid activation function: f(x) = max(0, min(1, 0.2x + 0.5))
__global__ void hard_sigmoid_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = 0.2f * input[idx] + 0.5f;
    if (val < 0.0f) val = 0.0f;
    else if (val > 1.0f) val = 1.0f;

    output[idx] = val;
}

int main() {
    const int size = 10;
    float h_input[size] = {-3, -2, -1, 0, 0.5, 1, 2, 3, 4, 5};
    float h_output[size];

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    hard_sigmoid_kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Hard Sigmoid Output:\n");
    for (int i = 0; i < size; ++i) {
        printf("Input: %.2f -> Output: %.4f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
