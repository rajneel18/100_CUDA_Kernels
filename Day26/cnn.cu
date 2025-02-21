#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h> // For fmaxf (ReLU)

// --- Helper functions (Error Checking) ---
#define CUDA_CHECK(call)                                                          \
    do {                                                                        \
        cudaError_t error = call;                                               \
        if (error != cudaSuccess) {                                             \
            printf("ERROR: %s:%d, ", __FILE__, __LINE__);                       \
            printf("code=%d, symbol=%s, message=%s\n", error,                   \
                   cudaGetErrorName(error), cudaGetErrorString(error));         \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// --- ReLU Activation Function (CUDA Kernel and Host Function) ---
__global__ void reluKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]); // ReLU: max(0, input[idx])
    }
}

void reluForward(float *input_d, float *output_d, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reluKernel<<<numBlocks, blockSize>>>(input_d, output_d, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --- ReLU Backward (Gradient) Kernel ---
__global__ void reluBackwardKernel(const float *output_grad, const float *output_forward, float *input_grad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < gridDim.x * blockDim.x) {
        if (output_forward[idx] > 0.0f) {
            input_grad[idx] = output_grad[idx]; // Pass gradient if input was positive
        } else {
            input_grad[idx] = 0.0f;             // Block gradient if input was zero or negative
        }
    }
}

void reluBackward(float *output_grad_d, float *output_forward_d, float *input_grad_d, int size) {
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reluBackwardKernel<<<numBlocks, blockSize>>>(output_grad_d, output_forward_d, input_grad_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// --- Img2col Unrolling Kernel and Host Function ---
__global__ void unrollKernel(const float *input, float *input_unrolled,
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

                    int input_idx = c * (input_height * input_width) +
                                     in_y * input_width + in_x;

                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        input_unrolled[unroll_idx] = input[input_idx];
                    } else {
                        // Padding (zero padding in this example, can be modified)
                        input_unrolled[unroll_idx] = 0.0f;
                    }
                }
            }
        }
    }
}


void unrollInput(float *input_d, float *input_unrolled_d,
                 int input_channels, int input_height, int input_width, int kernel_size,
                 int &output_height, int &output_width) {

    output_height = input_height - kernel_size + 1;
    output_width = input_width - kernel_size + 1;

    int output_size = output_height * output_width;

    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;

    unrollKernel<<<numBlocks, blockSize>>>(input_d, input_unrolled_d,
                                        input_channels, input_height, input_width,
                                        kernel_size, output_height, output_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


// --- Convolution Kernel (Matrix Multiplication with ReLU) ---
__global__ void convolutionKernel(const float *input_unrolled, const float *weights, float *output,
                                int output_size, int num_filters, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size * num_filters) {
        int filter_idx = idx / output_size;
        int output_pixel_idx = idx % output_size;

        float sum = 0.0f;
        for (int i = 0; i < filter_size; ++i) {
            sum += weights[filter_idx * filter_size + i] * input_unrolled[output_pixel_idx * filter_size + i];
        }
        output[idx] = sum; // Store the raw convolution output (before ReLU in this version)

    }
}


void convolutionForward(float *input_d, float *weights_d, float *output_d, int batch_size,
                      int num_filters, int input_channels, int input_height, int input_width, int kernel_size) {

    int output_height, output_width;
    int filter_size = input_channels * kernel_size * kernel_size;

    size_t unrolled_input_size = (size_t)(input_height - kernel_size + 1) * (input_width - kernel_size + 1) * filter_size * sizeof(float);
    float *input_unrolled_d;
    CUDA_CHECK(cudaMalloc(&input_unrolled_d, unrolled_input_size));

    int output_size = (input_height - kernel_size + 1) * (input_width - kernel_size + 1);

    for (int batch = 0; batch < batch_size; ++batch) {
        float *batch_input_d = input_d + batch * input_channels * input_height * input_width;
        float *batch_output_d = output_d + batch * num_filters * output_size;

        unrollInput(batch_input_d, input_unrolled_d, input_channels, input_height, input_width, kernel_size, output_height, output_width);

        int blockSize = 256;
        int numBlocks = ((output_size * num_filters) + blockSize - 1) / blockSize;

        convolutionKernel<<<numBlocks, blockSize>>>(input_unrolled_d, weights_d, batch_output_d, output_size, num_filters, filter_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Apply ReLU activation *after* convolution
        reluForward(batch_output_d, batch_output_d, output_size * num_filters); // In-place ReLU activation

    }
    CUDA_CHECK(cudaFree(input_unrolled_d));
}


// --- Backward Pass Kernels ---

__global__ void compute_dLdW(const float *dLdY, const float *input_unrolled, float *dLdW,
                             int output_height, int output_width, int num_filters, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_filters * filter_size) {
        int filter_idx = idx / filter_size;
        int weight_idx_in_filter = idx % filter_size;

        float sum_dW = 0.0f;
        for (int i = 0; i < output_height * output_width; ++i) {
            sum_dW += dLdY[filter_idx * (output_height * output_width) + i] * input_unrolled[i * filter_size + weight_idx_in_filter];
        }
        dLdW[idx] = sum_dW;
    }
}


__global__ void compute_dLdX(const float *dLdY, const float *weights, float *dLdX_unrolled,
                             int output_height, int output_width, int num_filters, int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_height * output_width * filter_size) {
        int output_pixel_idx = idx / filter_size;
        int input_channel_kx_ky_idx = idx % filter_size;

        float sum_dX = 0.0f;
        for (int i = 0; i < num_filters; ++i) {
            sum_dX += dLdY[i * (output_height * output_width) + output_pixel_idx] * weights[i * filter_size + input_channel_kx_ky_idx];
        }
        dLdX_unrolled[idx] = sum_dX;
    }
}


void convolutionBackward(int batch_size, int num_filters, int input_channels, int input_height, int input_width, int kernel_size,
                         float *dLdY_d, float *input_d, float *weights_d, float *dLdX_d, float *dLdW_d, float *output_forward_d) { //output_forward_d is the output from convolutionForward

    int output_height, output_width;
    int filter_size = input_channels * kernel_size * kernel_size;
    int output_size = (input_height - kernel_size + 1) * (input_width - kernel_size + 1);

    size_t unrolled_input_size = (size_t)output_size * filter_size * sizeof(float);
    float *input_unrolled_d;
    CUDA_CHECK(cudaMalloc(&input_unrolled_d, unrolled_input_size));

    size_t dLdX_unrolled_size = (size_t)output_size * filter_size * sizeof(float);
    float *dLdX_unrolled_d;
    CUDA_CHECK(cudaMalloc(&dLdX_unrolled_d, dLdX_unrolled_size));


    for (int batch = 0; batch < batch_size; ++batch) {
        float *batch_dLdY_d = dLdY_d + batch * num_filters * output_size;
        float *batch_input_d = input_d + batch * input_channels * input_height * input_width;
        float *batch_dLdX_d = dLdX_d + batch * input_channels * input_height * input_width;
        float *batch_output_forward_d = output_forward_d + batch * num_filters * output_size;


        // Backward pass for ReLU activation (assuming ReLU was used in forward)
        float *relu_input_grad_d = batch_dLdY_d; // In-place gradient for ReLU in this example
        reluBackward(batch_dLdY_d, batch_output_forward_d, relu_input_grad_d, output_size * num_filters);


        unrollInput(batch_input_d, input_unrolled_d, input_channels, input_height, input_width, kernel_size, output_height, output_width);

        int blockSize_dW = 256;
        int numBlocks_dW = ((num_filters * filter_size) + blockSize_dW - 1) / blockSize_dW;
        compute_dLdW<<<numBlocks_dW, blockSize_dW>>>(batch_dLdY_d, input_unrolled_d, dLdW_d, output_height, output_width, num_filters, filter_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());


        int blockSize_dX = 256;
        int numBlocks_dX = ((output_size * filter_size) + blockSize_dX - 1) / blockSize_dX;
        compute_dLdX<<<numBlocks_dX, blockSize_dX>>>(batch_dLdY_d, weights_d, dLdX_unrolled_d, output_height, output_width, num_filters, filter_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // "Roll back" dLdX_unrolled to dLdX (reshape and potentially sum if needed for deeper networks)
        // For simplicity, this example assumes dLdX has the same shape as input and does a direct copy (reshape)
        // In a full implementation, you'd need to handle padding, strides, and potential accumulation of gradients
        CUDA_CHECK(cudaMemcpy(batch_dLdX_d, dLdX_unrolled_d, input_channels * input_height * input_width * sizeof(float), cudaMemcpyDeviceToDevice));

    }

    CUDA_CHECK(cudaFree(input_unrolled_d));
    CUDA_CHECK(cudaFree(dLdX_unrolled_d));
}


// --- Max Pooling Backward Kernel ---
__global__ void maxPoolingBackwardKernel(const float *dLdY, const float *input, float *dLdX,
                                            int input_height, int input_width, int pool_size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < gridDim.x * blockDim.x) {
        int output_y = idx / ((input_width - pool_size) / stride + 1);
        int output_x = idx % ((input_width - pool_size) / stride + 1);

        float max_val = -999999.0f; // Initialize to a very small value
        int max_idx_y = -1, max_idx_x = -1;

        // Find the index of the maximum value in the pooling region (again - for backprop we need to know where max was in forward)
        for (int ky = 0; ky < pool_size; ++ky) {
            for (int kx = 0; kx < pool_size; ++kx) {
                int in_y = output_y * stride + ky;
                int in_x = output_x * stride + kx;
                int input_index = in_y * input_width + in_x;
                if (input[input_index] > max_val) {
                    max_val = input[input_index];
                    max_idx_y = in_y;
                    max_idx_x = in_x;
                }
            }
        }
        // Backpropagate gradient only to the max index
        int max_input_index = max_idx_y * input_width + max_idx_x;
        atomicAdd(&dLdX[max_input_index], dLdY[idx]); // Accumulate gradients using atomicAdd
    }
}


void maxPoolingBackward(float *dLdY_d, float *input_d, float *dLdX_d, int batch_size,
                        int input_channels, int input_height, int input_width, int pool_size, int stride) {

    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    int output_size = output_height * output_width;


    for (int batch = 0; batch < batch_size; ++batch) {
        float *batch_dLdY_d = dLdY_d + batch * input_channels * output_size;
        float *batch_input_d = input_d + batch * input_channels * input_height * input_width;
        float *batch_dLdX_d = dLdX_d + batch * input_channels * input_height * input_width;


        // Initialize dLdX to zero for accumulation
        CUDA_CHECK(cudaMemset(batch_dLdX_d, 0, input_channels * input_height * input_width * sizeof(float)));

        for (int channel = 0; channel < input_channels; ++channel) {
            float *channel_dLdY_d = batch_dLdY_d + channel * output_size;
            float *channel_input_d = batch_input_d + channel * input_height * input_width;
            float *channel_dLdX_d = batch_dLdX_d + channel * input_height * input_width;


            int blockSize = 256;
            int numBlocks = (output_size + blockSize - 1) / blockSize; // Launch threads for output size
            maxPoolingBackwardKernel<<<numBlocks, blockSize>>>(channel_dLdY_d, channel_input_d, channel_dLdX_d,
                                                                input_height, input_width, pool_size, stride);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}


// --- Test Function (Forward Pass Only for Convolutional Layer) ---
void testConvNet() {
    int batch_size = 1;
    int num_filters = 2;
    int input_channels = 1;
    int input_height = 4;
    int input_width = 4;
    int kernel_size = 3;

    float input_h[16] = {
        1, 1, 1, 0,
        0, 1, 1, 1,
        0, 0, 1, 1,
        0, 0, 0, 1
    };
    float weights_h[18] = { // 2 filters, 1 channel, 3x3 kernel
        1, 0, -1,
        1, 0, -1,
        1, 0, -1, // Filter 1
        1, 1, 0,
        0, 1, 1,
        0, 0, 1  // Filter 2
    };

    float *input_d, *weights_d, *output_d;

    CUDA_CHECK(cudaMalloc(&input_d, sizeof(input_h)));
    CUDA_CHECK(cudaMalloc(&weights_d, sizeof(weights_h)));
    size_t output_size_bytes = (size_t)batch_size * num_filters * (input_height - kernel_size + 1) * (input_width - kernel_size + 1) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&output_d, output_size_bytes));


    CUDA_CHECK(cudaMemcpy(input_d, input_h, sizeof(input_h), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weights_d, weights_h, sizeof(weights_h), cudaMemcpyHostToDevice));

    convolutionForward(input_d, weights_d, output_d, batch_size, num_filters, input_channels, input_height, input_width, kernel_size);

    float *output_h = (float*)malloc(output_size_bytes);
    CUDA_CHECK(cudaMemcpy(output_h, output_d, output_size_bytes, cudaMemcpyDeviceToHost));

    printf("Convolution Output with ReLU:\n");
    for (int f = 0; f < num_filters; ++f) {
        printf("Filter %d:\n", f + 1);
        for (int y = 0; y < input_height - kernel_size + 1; ++y) {
            for (int x = 0; x < input_width - kernel_size + 1; ++x) {
                printf("%.2f ", output_h[f * (input_height - kernel_size + 1) * (input_width - kernel_size + 1) + y * (input_width - kernel_size + 1) + x]);
            }
            printf("\n");
        }
    }


    free(output_h);
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(weights_d));
    CUDA_CHECK(cudaFree(output_d));
}


int main() {
    printf("Running CNN Convolution Example (CUDA C with ReLU)\n");
    testConvNet();
    printf("Finished CNN Convolution Example\n");
    return 0;
}