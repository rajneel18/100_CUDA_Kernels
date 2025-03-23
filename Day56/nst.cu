#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16 


__global__ void conv2D(float *input, float *kernel, float *output, int W, int H, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int halfK = K / 2;

    if (x >= W || y >= H) return;

    float sum = 0.0f;

    for (int i = -halfK; i <= halfK; i++) {
        for (int j = -halfK; j <= halfK; j++) {
            int newX = min(max(x + i, 0), W - 1);
            int newY = min(max(y + j, 0), H - 1);
            sum += input[newY * W + newX] * kernel[(i + halfK) * K + (j + halfK)];
        }
    }
    output[y * W + x] = sum;
}


__global__ void gramMatrix(float *features, float *gram, int C, int W, int H) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < C && col < C) {
        float sum = 0.0f;
        for (int i = 0; i < W * H; i++) {
            sum += features[row * W * H + i] * features[col * W * H + i];
        }
        gram[row * C + col] = sum;
    }
}


__global__ void updateImage(float *image, float *grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        image[idx] -= lr * grad[idx];
    }
}

void initializeData(float *data, int size, float value) {
    for (int i = 0; i < size; i++) {
        data[i] = value;
    }
}


void printArray(const float *data, int W, int H) {
    for (int i = 0; i < 5; i++) { 
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", data[i * W + j]);
        }
        printf("\n");
    }
    printf("...\n");
}

int main() {
    int W = 256, H = 256, K = 3, C = 64;
    float *input, *kernel, *output, *features, *gram, *grad;

    cudaMallocManaged(&input, W * H * sizeof(float));
    cudaMallocManaged(&kernel, K * K * sizeof(float));
    cudaMallocManaged(&output, W * H * sizeof(float));
    cudaMallocManaged(&features, C * W * H * sizeof(float));
    cudaMallocManaged(&gram, C * C * sizeof(float));
    cudaMallocManaged(&grad, W * H * sizeof(float));

    initializeData(input, W * H, 1.0f); 
    initializeData(kernel, K * K, 0.1f); 
    initializeData(features, C * W * H, 0.5f);
    initializeData(grad, W * H, 0.01f);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv2D<<<numBlocks, threadsPerBlock>>>(input, kernel, output, W, H, K);
    cudaDeviceSynchronize();
    printf("Convolution Output (First 5x5 Pixels):\n");
    printArray(output, W, H);

    gramMatrix<<<numBlocks, threadsPerBlock>>>(features, gram, C, W, H);
    cudaDeviceSynchronize();
    printf("Gram Matrix (First 5x5 Elements):\n");
    printArray(gram, C, C);

    updateImage<<<(W * H + 255) / 256, 256>>>(input, grad, 0.01, W * H);
    cudaDeviceSynchronize();
    printf("Updated Image (First 5x5 Pixels):\n");
    printArray(input, W, H);

    cudaFree(input);
    cudaFree(kernel);
    cudaFree(output);
    cudaFree(features);
    cudaFree(gram);
    cudaFree(grad);

    printf("Neural Style Transfer Completed on CUDA\n");
    return 0;
}
