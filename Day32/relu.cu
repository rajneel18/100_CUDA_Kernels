#include <stdio.h>
#include <cuda_runtime.h>

    // ReLU Kernel
__global__ void reluKernel(float *d_in, float *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = fmaxf(0.0f, d_in[idx]);
    }
}

int main() {
    int n = 1000;  
    size_t size = n * sizeof(float);
    
    float *h_in = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_in[i] = (i % 2 == 0) ? -i : i;
    }

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    reluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, n);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("ReLU(%f) = %f\n", h_in[i], h_out[i]);
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}