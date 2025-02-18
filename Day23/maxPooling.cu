#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// Kernel declaration - important!
__global__ void maxPoolingKernel_float(float* input, float* output, int input_height, int input_width, int pool_size, int stride);


__global__ void maxPoolingKernel_float(float* input, float* output, int input_height, int input_width, int pool_size, int stride) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        float max_value = -INFINITY;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int input_row = row * stride + i;
                int input_col = col * stride + j;
                if (input_row < input_height && input_col < input_width) {
                    max_value = fmaxf(max_value, input[input_row * input_width + input_col]);
                }
            }
        }
        output[row * output_width + col] = max_value;
    }
}


int main() {
    // Max Pooling Forward Test
    printf("\n--- Max Pooling Forward Test ---\n");
    int input_height = 4;
    int input_width = 4;
    int pool_size = 2;
    int stride = 2;
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;

    float input_h[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float *input_d, *output_d;
    float *output_h = (float*)malloc(output_height * output_width * sizeof(float));

    cudaMalloc(&input_d, sizeof(input_h));
    cudaMalloc(&output_d, output_height * output_width * sizeof(float));

    cudaMemcpy(input_d, input_h, sizeof(input_h), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, output_height * output_width * sizeof(float));

    dim3 maxPoolGridDim = dim3(output_width/16 + 1, output_height/16 + 1, 1);
    dim3 maxPoolBlockDim = dim3(16,16, 1);
    maxPoolingKernel_float<<<maxPoolGridDim, maxPoolBlockDim>>>(
        input_d, output_d, input_height, input_width, pool_size, stride);
    cudaDeviceSynchronize();

    cudaMemcpy(output_h, output_d, output_height * output_width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            printf("%.2f ", input_h[i * input_width + j]);
        }
        printf("\n");
    }

    printf("\nMax Pooling Output:\n");
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            printf("%.2f ", output_h[i * output_width + j]);
        }
        printf("\n");
    }

    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}