
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// Kernel and host function declarations
__global__ void reluBackwardKernel_float(float* dLdY, float* input, float* dLdX, int size);
void reluBackward(float* dLdY, float* input, float* dLdX, int size);


__global__ void reluBackwardKernel_float(float* dLdY, float* input, float* dLdX, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dLdX[idx] = (input[idx] > 0.0f) ? dLdY[idx] : 0.0f;
    }
}


void reluBackward(float* dLdY, float* input, float* dLdX, int size) {
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    reluBackwardKernel_float<<<blocksPerGrid, threadsPerBlock>>>(dLdY, input, dLdX, size);
    cudaDeviceSynchronize();
}


int main() {
    // ReLU Backward Test (dLdX)
    printf("\n--- ReLU Backward Test (dLdX) ---\n");
    int size = 8;
    float input_h[] = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
    float dLdY_h[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}; // Dummy dLdY
    float *input_d, *dLdY_d, *dLdX_d;
    float *dLdX_h = (float*)malloc(size * sizeof(float));

    cudaMalloc(&input_d, size * sizeof(float));
    cudaMalloc(&dLdY_d, size * sizeof(float));
    cudaMalloc(&dLdX_d, size * sizeof(float));

    cudaMemcpy(input_d, input_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dLdY_d, dLdY_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dLdX_d, 0, size * sizeof(float));

    reluBackward(dLdY_d, input_d, dLdX_d, size);
    cudaMemcpy(dLdX_h, dLdX_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("ReLU Input: ");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", input_h[i]);
    }
    printf("\n");
    printf("dLdY (ReLU Output Gradient): ");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", dLdY_h[i]);
    }
    printf("\n");

    printf("ReLU Backward dLdX Output: ");
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", dLdX_h[i]);
    }
    printf("\n");

    free(dLdX_h);
    cudaFree(input_d);
    cudaFree(dLdY_d);
    cudaFree(dLdX_d);

    return 0;
}
