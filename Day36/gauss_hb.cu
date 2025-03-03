#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Gaussian kernel function for smoothing
__device__ float gaussianKernel(float x, float mu, float sigma) {
    return expf(-0.5f * powf((x - mu) / sigma, 2.0f)) / (sigma * sqrtf(2.0f * M_PI));
}

// CUDA kernel to compute probability distribution using Gaussian kernel
__global__ void computeGaussianHistogram(float* data, int dataSize, 
                                         float* bins, int binCount,
                                         float minValue, float maxValue, 
                                         float bandwidth, float* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= binCount)
        return;
    
    float binCenter = bins[idx];
    float sum = 0.0f;
    
    // Apply Gaussian kernel to each data point for this bin
    for (int i = 0; i < dataSize; i++) {
        sum += gaussianKernel(data[i], binCenter, bandwidth);
    }
    
    // Normalize by the number of data points
    result[idx] = sum / dataSize;
}

// Helper function to linearly space bin centers
void linspace(float min, float max, int count, float* result) {
    float step = (max - min) / (count - 1);
    for (int i = 0; i < count; i++) {
        result[i] = min + i * step;
    }
}

int main(int argc, char** argv) {
    int dataSize = 10000;
    int binCount = 100;
    float minValue = -5.0f;
    float maxValue = 5.0f;
    float bandwidth = 0.5f;
    
    float* h_data = (float*)malloc(dataSize * sizeof(float));
    float* h_bins = (float*)malloc(binCount * sizeof(float));
    float* h_result = (float*)malloc(binCount * sizeof(float));
    
    srand(42);
    for (int i = 0; i < dataSize; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        h_data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
    
    linspace(minValue, maxValue, binCount, h_bins);
    
    float *d_data, *d_bins, *d_result;
    cudaMalloc(&d_data, dataSize * sizeof(float));
    cudaMalloc(&d_bins, binCount * sizeof(float));
    cudaMalloc(&d_result, binCount * sizeof(float));
    
    cudaMemcpy(d_data, h_data, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins, h_bins, binCount * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (binCount + threadsPerBlock - 1) / threadsPerBlock;
    
    computeGaussianHistogram<<<blocksPerGrid, threadsPerBlock>>>(
        d_data, dataSize, d_bins, binCount, minValue, maxValue, bandwidth, d_result);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    cudaMemcpy(h_result, d_result, binCount * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Bin Center | Probability Density\n");
    printf("-------------------------------\n");
    for (int i = 0; i < binCount; i++) {
        printf("%9.2f | %f\n", h_bins[i], h_result[i]);
    }
    
    free(h_data);
    free(h_bins);
    free(h_result);
    cudaFree(d_data);
    cudaFree(d_bins);
    cudaFree(d_result);
    
    return 0;
}