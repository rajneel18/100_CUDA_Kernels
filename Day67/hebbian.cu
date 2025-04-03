#include <stdio.h>
#include <cuda_runtime.h>

#define N 256    // Number of neurons
#define ETA 0.01 // Learning rate

// CUDA Kernel for Hebbian Learning Rule
__global__ void hebbian_update(float* weights, float* inputs, float* outputs, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < size && j < size) {
        int idx = i * size + j;
        weights[idx] += ETA * inputs[i] * outputs[j]; // Hebbian update rule
    }
}

int main() {
    // Allocate host memory
    float *h_inputs, *h_outputs, *h_weights;
    h_inputs = (float*)malloc(N * sizeof(float));
    h_outputs = (float*)malloc(N * sizeof(float));
    h_weights = (float*)malloc(N * N * sizeof(float));

    // Initialize inputs, outputs, and weights
    for (int i = 0; i < N; i++) {
        h_inputs[i] = (float)rand() / RAND_MAX;
        h_outputs[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < N * N; i++) {
        h_weights[i] = 0.0f;
    }

    // Allocate device memory
    float *d_inputs, *d_outputs, *d_weights;
    cudaMalloc(&d_inputs, N * sizeof(float));
    cudaMalloc(&d_outputs, N * sizeof(float));
    cudaMalloc(&d_weights, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_inputs, h_inputs, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, h_outputs, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (N + 15) / 16);

    // Run Hebbian learning kernel
    hebbian_update<<<gridSize, blockSize>>>(d_weights, d_inputs, d_outputs, N);
    cudaDeviceSynchronize();

    // Copy updated weights back to host
    cudaMemcpy(h_weights, d_weights, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few updated weights
    printf("Updated weights (first 5 values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_weights[i]);
    }
    printf("\n");

    // Free memory
    free(h_inputs); free(h_outputs); free(h_weights);
    cudaFree(d_inputs); cudaFree(d_outputs); cudaFree(d_weights);

    return 0;
}
