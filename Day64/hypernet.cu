#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define INPUT_SIZE 3
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 2

#define GENERATED_WEIGHTS (HIDDEN_SIZE * OUTPUT_SIZE)

// CUDA Kernel: Generate weights dynamically
__global__ void generate_weights(float *weights, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < GENERATED_WEIGHTS) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_uniform(&state) * 2.0f - 1.0f;  // Random values in [-1, 1]
    }
}

// CUDA Kernel: Forward pass for the main network
__global__ void forward_pass(float *input, float *weights, float *output) {
    int idx = threadIdx.x;
    if (idx < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            sum += input[i] * weights[i * OUTPUT_SIZE + idx];
        }
        output[idx] = tanhf(sum);  // Apply activation function
    }
}

int main() {
    // Allocate memory
    float *d_weights, *d_input, *d_output;
    float h_input[HIDDEN_SIZE] = {1.0f, -0.5f, 0.8f, -1.2f};  // Example input
    float h_output[OUTPUT_SIZE];

    cudaMalloc(&d_weights, GENERATED_WEIGHTS * sizeof(float));
    cudaMalloc(&d_input, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Generate dynamic weights using CUDA
    generate_weights<<<1, GENERATED_WEIGHTS>>>(d_weights, 42);
    cudaDeviceSynchronize();

    // Perform forward pass using dynamically generated weights
    forward_pass<<<1, OUTPUT_SIZE>>>(d_input, d_weights, d_output);
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    printf("Network Output: [%f, %f]\n", h_output[0], h_output[1]);

    // Free memory
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
