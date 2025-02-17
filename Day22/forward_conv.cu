#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

__global__ void unrollKernel_float(const float* input, float* input_unrolled,
    int input_channels, int input_height, int input_width,
    int kernel_size, int output_height, int output_width) {
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
                    int input_idx = c * (input_height * input_width) + in_y * input_width + in_x;

                    input_unrolled[unroll_idx] = input[input_idx];
                }
            }
        }
    }
}

__global__ void convolutionKernel_float(const float* input_unrolled, const float* weights, const float* bias, float* output,
    int output_size, int num_filters, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size * num_filters) {
        int output_idx = idx / num_filters;
        int filter_idx = idx % num_filters;

        float sum = 0.0f;
        for (int i = 0; i < filter_size; i++) {
            sum += input_unrolled[output_idx * filter_size + i] * weights[filter_idx * filter_size + i];
        }
        output[idx] = sum + bias[filter_idx];
    }
}

void convolutionForward(float* input, float* weights, float* bias, float* output,
    int batch_size, int num_filters, int input_channels,
    int input_height, int input_width, int kernel_size) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int output_size = output_height * output_width;
    int filter_size = input_channels * kernel_size * kernel_size;

    float* input_unrolled;
    cudaMalloc(&input_unrolled, output_size * filter_size * sizeof(float));

    int unroll_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int conv_blocks = (output_size * num_filters + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unrollKernel_float<<<unroll_blocks, BLOCK_SIZE>>>(input, input_unrolled, input_channels, input_height, input_width, kernel_size, output_height, output_width);
    convolutionKernel_float<<<conv_blocks, BLOCK_SIZE>>>(input_unrolled, weights, bias, output, output_size, num_filters, filter_size);

    cudaFree(input_unrolled);
}

int main() {
    printf("\n--- Convolution Forward Test ---\n");
    int batch_size = 1, num_filters = 2, input_channels = 1, input_height = 4, input_width = 4, kernel_size = 2;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int output_size = output_height * output_width;
    int filter_size = input_channels * kernel_size * kernel_size;

    float input_h[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    float weights_h[] = { 1, 2, 3, 4, -1, -2, -3, -4 };
    float bias_h[] = {0.5f, -0.5f};
    float *input_d, *weights_d, *bias_d, *output_d;
    float *output_h = (float*)malloc(batch_size * num_filters * output_size * sizeof(float));

    cudaMalloc(&input_d, sizeof(input_h));
    cudaMalloc(&weights_d, sizeof(weights_h));
    cudaMalloc(&bias_d, sizeof(bias_h));
    cudaMalloc(&output_d, batch_size * num_filters * output_size * sizeof(float));

    cudaMemcpy(input_d, input_h, sizeof(input_h), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d, weights_h, sizeof(weights_h), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias_h, sizeof(bias_h), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0, batch_size * num_filters * output_size * sizeof(float));

    convolutionForward(input_d, weights_d, bias_d, output_d, batch_size, num_filters, input_channels, input_height, input_width, kernel_size);
    cudaMemcpy(output_h, output_d, batch_size * num_filters * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input:\n");
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < input_width; ++j) {
            printf("%.2f ", input_h[i * input_width + j]);
        }
        printf("\n");
    }

    printf("\nWeights:\n");
    for (int i = 0; i < 2 * filter_size; ++i) {
        printf("%.2f ", weights_h[i]);
        if ((i + 1) % filter_size == 0) printf("  ");
    }
    printf("\n\nConvolution Output:\n");
    for (int f = 0; f < num_filters; ++f) {
        printf("Filter %d:\n", f);
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                printf("%.2f ", output_h[f * output_size + i * output_width + j]);
            }
            printf("\n");
        }
    }

    free(output_h);
    cudaFree(input_d);
    cudaFree(weights_d);
    cudaFree(bias_d);
    cudaFree(output_d);
    return 0;
}
