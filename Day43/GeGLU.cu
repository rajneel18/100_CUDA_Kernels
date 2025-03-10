#include <cmath>
#include <cuda_runtime.h>
#include <stdio.h>

// Define GELU function
__device__ float gelu(float x) {
    return 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI) * (x + 0.044715f * pow(x, 3))));
}

// GEGLU CUDA kernel
__global__ void geglu_kernel(float *input, float *W1, float *W2, float *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if we are within bounds
    if (idx < size) {
        // Apply GELU activation to W1 * input
        float W1_x = W1[idx] * input[idx];
        float W2_x = W2[idx] * input[idx];
        
        // Apply GELU to W1 * input and multiply element-wise with W2 * input
        output[idx] = gelu(W1_x) * W2_x;
    }
}

void geglu(float *input, float *W1, float *W2, float *output, int size) {
    float *d_input, *d_W1, *d_W2, *d_output;


    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_W1, size * sizeof(float));
    cudaMalloc(&d_W2, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    
    cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, size * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    geglu_kernel<<<numBlocks, blockSize>>>(d_input, d_W1, d_W2, d_output, size);

    cudaMemcpy(output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_output);
}

int main() {
    const int size = 1024;  // Example size of the input
    float input[size], W1[size], W2[size], output[size];

    // Initialize input and weights
    for (int i = 0; i < size; i++) {
        input[i] = static_cast<float>(rand()) / RAND_MAX; // Random input between 0 and 1
        W1[i] = 1.0f;  // Example weight
        W2[i] = 1.0f;  // Example weight
    }

    // Call the GEGLU function
    geglu(input, W1, W2, output, size);

    // Optionally, print the output for debugging
    for (int i = 0; i < 10; i++) {  // Print first 10 values for example
        printf("output[%d] = %f\n", i, output[i]);
    }

    return 0;
}