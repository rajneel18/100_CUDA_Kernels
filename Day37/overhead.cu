#include <stdio.h>
#include <cuda_runtime.h>

#define N 500000    
#define NSTEP 1000  
#define NKERNEL 20  

#define THREADS_PER_BLOCK 512

__global__ void shortKernel(float *out_d, float *in_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out_d[idx] = 1.23f * in_d[idx];
    }
}

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(result));
        exit(-1);
    }
}

int main() {
    float *d_in, *d_out;
    cudaStream_t stream;
    
    checkCuda(cudaMalloc(&d_in, N * sizeof(float)), "Alloc d_in");
    checkCuda(cudaMalloc(&d_out, N * sizeof(float)), "Alloc d_out");

    checkCuda(cudaStreamCreate(&stream), "Stream creation");

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Event start creation");
    checkCuda(cudaEventCreate(&stop), "Event stop creation");
    checkCuda(cudaEventRecord(start), "Start event recording");

    for (int istep = 0; istep < NSTEP; istep++) {
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
            shortKernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(d_out, d_in);
            cudaStreamSynchronize(stream);
        }
    }

    checkCuda(cudaEventRecord(stop), "Stop event recording");
    checkCuda(cudaEventSynchronize(stop), "Stop event synchronization");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Elapsed time calculation");

    printf("Time taken per kernel: %f microseconds\n", (milliseconds * 1000) / (NSTEP * NKERNEL));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
