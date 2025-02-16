#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

__global__ void unrollKernel_float(const float* input, float* input_unrolled,
    const int input_channels, const int input_height, const int input_width,
    const int kernel_size, const int output_height, const int output_width); // Kernel declaration - important!

void unrollInput(int input_channels, int input_height, int input_width,
    int kernel_size, float* input, float* input_unrolled); // Host function declaration - important!


__global__ void unrollKernel_float(const float* input, float* input_unrolled,
    const int input_channels, const int input_height, const int input_width,
    const int kernel_size, const int output_height, const int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = output_height * output_width;

    if (idx < total_elements) {
        int out_y = idx / output_width;
        int out_x = idx % output_width;

        for (int c = 0; c < input_channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;

                    int unroll_idx = idx * (input_channels * kernel_size * kernel_size) +
                        (c * kernel_size * kernel_size + ky * kernel_size + kx);

                    int input_idx = c * (input_height * input_width) +
                        in_y * input_width + in_x;

                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        input_unrolled[unroll_idx] = input[input_idx];
                    } else {
                        input_unrolled[unroll_idx] = 0.0f;
                    }
                }
            }
        }
    }
}

void unrollInput(int input_channels, int input_height, int input_width,
    int kernel_size, float* input, float* input_unrolled) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int total_output_elements = output_height * output_width;

    int threadsPerBlock = 256;
    int numBlocks = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;

    unrollKernel_float<<<numBlocks, threadsPerBlock>>>(
        input,
        input_unrolled,
        input_channels,
        input_height,
        input_width,
        kernel_size,
        output_height,
        output_width
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in unroll: %s\n", cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
}


int main() {
    // Unrolling Test
    printf("--- Unrolling Test ---\n");
    int input_channels = 1;
    int input_height = 4;
    int input_width = 4;
    int kernel_size = 2;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;
    int output_size = output_height * output_width;

    float input_h[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float *input_d, *input_unrolled_d;
    float *input_unrolled_h = (float*)malloc(output_size * filter_size * sizeof(float));

    cudaMalloc(&input_d, sizeof(input_h));
    cudaMalloc(&input_unrolled_d, output_size * filter_size * sizeof(float));
    cudaMemcpy(input_d, input_h, sizeof(input_h), cudaMemcpyHostToDevice);


    unrollInput(input_channels, input_height, input_width, kernel_size, input_d, input_unrolled_d);
    cudaMemcpy(input_unrolled_h, input_unrolled_d, output_size * filter_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            printf("%.2f ", input_h[i * input_width + j]);
        }
        printf("\n");
    }

    printf("\nUnrolled Input:\n");
    for (int i = 0; i < output_size; ++i) {
        printf("Output Pixel %d: ", i);
        for (int j = 0; j < filter_size; ++j) {
            printf("%.2f ", input_unrolled_h[i * filter_size + j]);
        }
        printf("\n");
    }

    free(input_unrolled_h);
    cudaFree(input_d);
    cudaFree(input_unrolled_d);

    return 0;
}