#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5

__global__ void layer_norm(float* input, float* output, float* gamma, float* beta, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= D) return;

    // mean
    float mean = 0.0f;
    for (int i = 0; i < D; i++) {
        mean += input[i];
    }
    mean /= D;

    // variance
    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        var += (input[i] - mean) * (input[i] - mean);
    }
    var /= D;

    //  Normalize and scale-shift
    output[idx] = gamma[idx] * ((input[idx] - mean) / sqrtf(var + EPSILON)) + beta[idx];
}

void layerNormHost(float* input, float* output, float* gamma, float* beta, int D) {
    float *d_input, *d_output, *d_gamma, *d_beta;
    size_t size = D * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_gamma, size);
    cudaMalloc(&d_beta, size);

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, size, cudaMemcpyHostToDevice);

    layer_norm<<<1, D>>>(d_input, d_output, d_gamma, d_beta, D);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

int main() {
    int D = 5;
    float h_input[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_gamma[5] = {1.0, 1.0, 1.0, 1.0, 1.0}; 
    float h_beta[5] = {0.0, 0.0, 0.0, 0.0, 0.0};  
    float h_output[5];

    layerNormHost(h_input, h_output, h_gamma, h_beta, D);

    printf("Layer Normalized Output:\n");
    for (int i = 0; i < D; i++) {
        printf("%f\n", h_output[i]);
    }

    return 0;
}
