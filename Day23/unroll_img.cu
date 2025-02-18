#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// Include stb_image for image loading
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
    // Unrolling Test with Image Input
    printf("--- Unrolling Test with Image Input ---\n");

    // Image loading using stb_image
    int input_width, input_height, input_channels;
    float *input_h;
    const char *image_path = "input_image.jpg"; // Path to your input image
    int kernel_size = 2;

    // Load image - assuming grayscale for simplicity. You might need to adapt for color images.
    unsigned char *image_data = stbi_load(image_path, &input_width, &input_height, &input_channels, 1); // Load as grayscale
    if (image_data == NULL) {
        printf("Error loading image: %s\n", image_path);
        return 1;
    }
    input_channels = 1; // Force single channel since we're loading as grayscale
    printf("Loaded image: %s, Width: %d, Height: %d, Channels: %d (Grayscale)\n", image_path, input_width, input_height, input_channels);


    // Convert image data to float and normalize if needed (e.g., to 0-1 range)
    input_h = (float*)malloc(input_height * input_width * sizeof(float));
    if (input_h == NULL) {
        printf("Host memory allocation failed!\n");
        stbi_image_free(image_data);
        return 1;
    }
    for (int i = 0; i < input_height * input_width; ++i) {
        input_h[i] = (float)image_data[i] / 255.0f; // Normalize to 0-1 range
    }
    stbi_image_free(image_data); // Free image data after copying to float array


    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;
    int output_size = output_height * output_width;

    float *input_d, *input_unrolled_d;
    float *input_unrolled_h = (float*)malloc(output_size * filter_size * sizeof(float));
    if (input_unrolled_h == NULL) {
        printf("Host output memory allocation failed!\n");
        free(input_h);
        return 1;
    }

    cudaError_t cuda_status;
    cuda_status = cudaMalloc(&input_d, input_height * input_width * sizeof(float));
    if (cuda_status!= cudaSuccess) {
        printf("cudaMalloc input_d failed! %s\n", cudaGetErrorString(cuda_status));
        free(input_h);
        free(input_unrolled_h);
        return 1;
    }

    cuda_status = cudaMalloc(&input_unrolled_d, output_size * filter_size * sizeof(float));
    if (cuda_status!= cudaSuccess) {
        printf("cudaMalloc input_unrolled_d failed! %s\n", cudaGetErrorString(cuda_status));
        free(input_h);
        free(input_unrolled_h);
        cudaFree(input_d);
        return 1;
    }
    cudaMemset(input_unrolled_d, 0, output_size * filter_size * sizeof(float));


    cuda_status = cudaMemcpy(input_d, input_h, input_height * input_width * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_status!= cudaSuccess) {
        printf("cudaMemcpy input_d failed! %s\n", cudaGetErrorString(cuda_status));
        free(input_h);
        free(input_unrolled_h);
        cudaFree(input_d);
        cudaFree(input_unrolled_d);
        return 1;
    }


    unrollInput(input_channels, input_height, input_width, kernel_size, input_d, input_unrolled_d);
    cuda_status = cudaMemcpy(input_unrolled_h, input_unrolled_d, output_size * filter_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_status!= cudaSuccess) {
        printf("cudaMemcpy output_d failed! %s\n", cudaGetErrorString(cuda_status));
        free(input_h);
        free(input_unrolled_h);
        cudaFree(input_d);
        cudaFree(input_unrolled_d);
        return 1;
    }

    printf("Input Image (first few pixels):\n");
    for (int i = 0; i < input_height && i < 4; ++i) {
        for (int j = 0; j < input_width && j < 4; ++j) {
            printf("%.2f ", input_h[i * input_width + j]);
        }
        printf("...\n");
        if (input_width > 4) break; // To avoid printing very long lines for wide images
    }
    if (input_height > 4) printf("...\n\n");
    else printf("\n\n");


    printf("Unrolled Input (first few output pixels):\n");
    for (int i = 0; i < output_size && i < 4; ++i) {
        printf("Output Pixel %d: ", i);
        for (int j = 0; j < filter_size; ++j) {
            printf("%.2f ", input_unrolled_h[i * filter_size + j]);
        }
        printf("\n");
    }
    if (output_size > 4) printf("...\n");


    free(input_h);
    free(input_unrolled_h);
    cudaFree(input_d);
    cudaFree(input_unrolled_d);

    return 0;
}