#include <stdio.h>
#include <cuda_runtime.h>

#define N 500000     
#define NSTEP 1000   
#define NKERNEL 20   

// CUDA Kernel
__global__ void shortKernel(float *out_d, float *in_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out_d[idx] = 1.23f * in_d[idx];
}

// Error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    float *in_d, *out_d;
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    
    CHECK_CUDA(cudaMalloc(&in_d, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_d, N * sizeof(float)));


    CHECK_CUDA(cudaStreamCreate(&stream));

    int threads, minGridSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threads, shortKernel, 0, N));
    int blocks = (N + threads - 1) / threads;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
        shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }

    // create graph
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // Warm-up Execution
    CHECK_CUDA(cudaGraphLaunch(instance, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    float elapsedTime;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));

    for (int istep = 0; istep < NSTEP; istep++) {
        CHECK_CUDA(cudaGraphLaunch(instance, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));

    float timePerKernel = (elapsedTime * 1000) / (NSTEP * NKERNEL);
    printf("Time taken per kernel using CUDA Graphs: %.3f microseconds\n", timePerKernel);

    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaGraphExecDestroy(instance));
    CHECK_CUDA(cudaFree(in_d));
    CHECK_CUDA(cudaFree(out_d));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
