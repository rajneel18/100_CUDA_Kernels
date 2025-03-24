#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 256  // CUDA Block size

__global__ void computeGradient(float *weights, float *grads, float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grads[idx] = 2 * (data[idx] - weights[idx]); // Example: dL/dW = 2 * (y_pred - y)
    }
}

__global__ void computeHessian(float *grads, float *hessian, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hessian[idx] = fabsf(grads[idx]);  // Approximate Hessian as abs(gradient)
    }
}

__global__ void updateWeights(float *weights, float *grads, float *hessian, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ada_hessian_term = 1.0f / (sqrtf(hessian[idx]) + 1e-4f);  // AdaHessian scaling
        weights[idx] -= lr * grads[idx] * ada_hessian_term;
    }
}

int main() {
    int size = 1024;  // Number of parameters
    float lr = 0.01;  // Learning rate

    float *weights, *grads, *hessian, *data;

    cudaMallocManaged(&weights, size * sizeof(float));
    cudaMallocManaged(&grads, size * sizeof(float));
    cudaMallocManaged(&hessian, size * sizeof(float));
    cudaMallocManaged(&data, size * sizeof(float));
    

    for (int i = 0; i < size; i++) {
        weights[i] = 0.5f; // Example: Starting weight
        data[i] = 1.0f;    // Target values
    }

    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int epoch = 0; epoch < 100; epoch++) {  // Training loop
        computeGradient<<<numBlocks, BLOCK_SIZE>>>(weights, grads, data, size);
        cudaDeviceSynchronize();

        computeHessian<<<numBlocks, BLOCK_SIZE>>>(grads, hessian, size);
        cudaDeviceSynchronize();

        updateWeights<<<numBlocks, BLOCK_SIZE>>>(weights, grads, hessian, lr, size);
        cudaDeviceSynchronize();
    }

    printf("Final weight[0]: %f\n", weights[0]);

    cudaFree(weights);
    cudaFree(grads);
    cudaFree(hessian);
    cudaFree(data);

    return 0;
}
