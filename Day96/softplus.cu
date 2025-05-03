#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float softplus(float x) {
    return logf(1.0f + expf(x));
}

__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = softplus(input[idx]);
    }
}

int main() {
    const int size = 8;
    float h_input[size] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_output[size];

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    softplus_kernel<<<blocks, threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Softplus output:\n");
    for (int i = 0; i < size; ++i) {
        printf("softplus(%.2f) = %.6f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
