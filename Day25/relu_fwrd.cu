#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// Kernel and host function declarations
__global__ void reluKernel_float(float* input, float* output, int size);
void reluForward(float* input, float* output, int size);


__global__ void reluKernel_float(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}


void reluForward(float* input, float* output, int size) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel_float<<<blocksPerGrid, threadsPerBlock>>>(input, output, size);
    cudaDeviceSynchronize();
}


int main() {
    // ReLU Forward Test
    printf("\n--- ReLU Forward Test ---\n");
    int size = 8;
    float input_h[] = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
    float *input_d, *output_d;
    float *output_h = (float*)malloc(size * sizeof(float));

    cudaMalloc(&input_d, size * sizeof(float));
    cudaMalloc(&output_d, size * sizeof(float));

    cudaMemcpy(input_d, input_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, size * sizeof(float));

    reluForward(input_d, output_d, size);
    cudaMemcpy(output_h, output_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input: ");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", input_h[i]);
    }
    printf("\n");

    printf("ReLU Output: ");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", output_h[i]);
    }
    printf("\n");

    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}