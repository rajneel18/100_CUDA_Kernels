#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

__global__ void FHT_kernel(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float angle = (2.0f * PI * idx * k) / N;
            sum += input[k] * (cos(angle) + sin(angle));
        }
        output[idx] = sum;
    }
}

void computeFHT(float* h_input, float* h_output, int N) {
    float *d_input, *d_output;
    
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    
    FHT_kernel<<<numBlocks, blockSize>>>(d_input, d_output, N);
    
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const int N = 1024; 
    float* h_input = new float[N];
    float* h_output = new float[N];
    
    // Initialize input array
    for (int i = 0; i < N; i++) {
        h_input[i] = sinf(2.0f * PI * i / N); 
    }
    
    // Compute FHT
    computeFHT(h_input, h_output, N);
    

    printf("First 10 FHT coefficients:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");
    
    delete[] h_input;
    delete[] h_output;
    
    return 0;
}