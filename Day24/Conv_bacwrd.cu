#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// Kernel and host function declarations
__global__ void unrollKernel_float(const float* input, float* input_unrolled,
    const int input_channels, const int input_height, const int input_width,
    const int kernel_size, const int output_height, const int output_width);
void unrollInput(int input_channels, int input_height, int input_width,
    int kernel_size, float* input, float* input_unrolled);
__global__ void compute_dLdW_float(float* dLdY, float* input_unrolled, float* dLdW, int output_height, int output_width, int num_filters, int filter_size);
__global__ void compute_dLdB_float(float* dLdY, float* dLdB, int output_height, int output_width, int num_filters);
void convolutionBackward(int batch_size, int num_filters, int input_channels, int input_height, int input_width, int kernel_size, float* dLdY, float* input, float* weights, float* bias, float* dLdX, float* dLdW, float* dLdB);


__global__ void unrollKernel_float(const float* input, float* input_unrolled,
    const int input_channels, const int input_height, const int input_width,
    const int kernel_size, const int output_height, const int output_width) {
    // ... (unrollKernel_float implementation from Part 1) ...
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
    // ... (unrollInput implementation from Part 1) ...
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


__global__ void compute_dLdW_float(float* dLdY, float* input_unrolled, float* dLdW, int output_height, int output_width, int num_filters, int filter_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < filter_size && col < num_filters) {
        float sum = 0;
        for (int i = 0; i < output_height * output_width; i++) {
            sum += input_unrolled[i * filter_size + row] * dLdY[i * num_filters + col];
        }
        dLdW[row * num_filters + col] = sum;
    }
}

__global__ void compute_dLdB_float(float* dLdY, float* dLdB, int output_height, int output_width, int num_filters) {
    int filter_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (filter_idx < num_filters) {
        float sum = 0;
        for (int i = 0; i < output_height * output_width; ++i) {
            sum += dLdY[i * num_filters + filter_idx];
        }
        dLdB[filter_idx] = sum;
    }
}


void convolutionBackward(int batch_size, int num_filters, int input_channels, int input_height, int input_width, int kernel_size, float* dLdY, float* input, float* weights, float* bias, float* dLdX, float* dLdW, float* dLdB) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;

    float* input_unrolled;
    float* dLdX_unrolled;
    cudaMalloc(&input_unrolled, output_height * output_width * filter_size * sizeof(float));
    cudaMalloc(&dLdX_unrolled, output_height * output_width * filter_size * sizeof(float));
    cudaMemset(dLdX, 0, batch_size * input_channels * input_height * input_width * sizeof(float));
    cudaMemset(dLdW, 0, num_filters * filter_size * sizeof(float));
    cudaMemset(dLdB, 0, num_filters * sizeof(float));


    for (int n = 0; n < batch_size; n++) {
        unrollInput(input_channels, input_height, input_width, kernel_size, input + n * input_channels * input_height * input_width, input_unrolled);

        dim3 blockSize_dLdW(16, 16);
        dim3 gridSize_dLdW((num_filters + blockSize_dLdW.x - 1) / blockSize_dLdW.x, (filter_size + blockSize_dLdW.y - 1) / blockSize_dLdW.y);
        dim3 blockSize_dLdB(256);
        dim3 gridSize_dLdB((num_filters + blockSize_dLdB.x - 1) / blockSize_dLdB.x);


        compute_dLdW_float<<<gridSize_dLdW, blockSize_dLdW>>>(dLdY + n * num_filters * output_height * output_width, input_unrolled, dLdW, output_height, output_width, num_filters, filter_size);
        compute_dLdB_float<<<gridSize_dLdB, blockSize_dLdB>>>(dLdY + n * num_filters * output_height * output_width, dLdB, output_height, output_width, num_filters);


        cudaDeviceSynchronize();
    }

    cudaFree(input_unrolled);
    cudaFree(dLdX_unrolled);
}


int main() {
    // Convolution Backward Gradients Test (dLdW and dLdB)
    printf("\n--- Convolution Backward Gradients Test (dLdW and dLdB) ---\n");
    int batch_size = 1;
    int num_filters = 2;
    int input_channels = 1;
    int input_height = 4;
    int input_width = 4;
    int kernel_size = 2;
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;

    float input_h[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    float weights_h[] = {
        1, 2, 3, 4,  // Filter 1 (not used in backward test directly, but needed in function signature)
        -1, -2, -3, -4 // Filter 2 (not used in backward test directly, but needed in function signature)
    };

    float bias_h[] = {0.5f, -0.5f}; // Bias for two filters (not used in backward test directly, but needed in function signature)


    float dLdY_h[] = { // Dummy dLdY
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    float *input_d, *weights_d, *bias_d, *dLdY_d;
    float *dLdW_d, *dLdB_d, *dLdX_d; // dLdX is not tested here, just needed for function call
    float *dLdW_h = (float*)malloc(num_filters * filter_size * sizeof(float));
    float *dLdB_h = (float*)malloc(num_filters * sizeof(float));
    float *dLdX_h = (float*)malloc(batch_size * input_channels * input_height * input_width * sizeof(float)); //dummy, not used


    cudaMalloc(&input_d, sizeof(input_h));
    cudaMalloc(&weights_d, sizeof(weights_h));
    cudaMalloc(&bias_d, sizeof(bias_h));
    cudaMalloc(&dLdY_d, sizeof(dLdY_h));
    cudaMalloc(&dLdW_d, num_filters * filter_size * sizeof(float));
    cudaMalloc(&dLdB_d, num_filters * sizeof(float));
    cudaMalloc(&dLdX_d, batch_size * input_channels * input_height * input_width * sizeof(float)); //dummy

    cudaMemcpy(input_d, input_h, sizeof(input_h), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d, weights_h, sizeof(weights_h), cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, bias_h, sizeof(bias_h), cudaMemcpyHostToDevice);
    cudaMemcpy(dLdY_d, dLdY_h, sizeof(dLdY_h), cudaMemcpyHostToDevice);
    cudaMemset(dLdW_d, 0, num_filters * filter_size * sizeof(float));
    cudaMemset(dLdB_d, 0, num_filters * sizeof(float));
    cudaMemset(dLdX_d, 0, batch_size * input_channels * input_height * input_width * sizeof(float)); //dummy

    convolutionBackward(batch_size, num_filters, input_channels, input_height, input_width, kernel_size, dLdY_d, input_d, weights_d, bias_d, dLdX_d, dLdW_d, dLdB_d);

    cudaMemcpy(dLdW_h, dLdW_d, num_filters * filter_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dLdB_h, dLdB_d, num_filters * sizeof(float), cudaMemcpyDeviceToHost);


    printf("dLdW Output:\n");
    for (int f = 0; f < num_filters; f++) {
        printf("Filter %d:\n", f);
        for (int i = 0; i < filter_size; i++) {
                printf("%.2f ", dLdW_h[f * filter_size + i]);
        }
        printf("\n");
    }

    printf("dLdB Output:\n");
    for (int f = 0; f < num_filters; f++) {
        printf("Filter %d: %.2f\n", f, dLdB_h[f]);
    }


    free(dLdW_h);
    free(dLdB_h);
    free(dLdX_h); //dummy
    cudaFree(input_d);
    cudaFree(weights_d);
    cudaFree(bias_d);
    cudaFree(dLdY_d);
    cudaFree(dLdW_d);
    cudaFree(dLdB_d);
    cudaFree(dLdX_d); //dummy

    return 0;
}