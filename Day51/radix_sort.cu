#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 16  // Adjust for larger data sizes
#define BLOCK_SIZE 256
#define RADIX 10  // Decimal system radix

// CUDA error check macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel to compute digit frequencies (histogram)
__global__ void countSortKernel(int *d_input, int *d_output, int *hist, int exp, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        int digit = (d_input[tid] / exp) % RADIX;
        atomicAdd(&hist[digit], 1);  // Atomic increment histogram
    }
}

// Kernel to perform prefix sum (scan) for histogram
__global__ void scanKernel(int *hist, int *scan_hist) {
    __shared__ int temp[RADIX];
    
    int tid = threadIdx.x;
    if (tid < RADIX) temp[tid] = hist[tid];
    __syncthreads();
    
    for (int stride = 1; stride < RADIX; stride *= 2) {
        int val = 0;
        if (tid >= stride) val = temp[tid - stride];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    
    if (tid < RADIX) scan_hist[tid] = temp[tid];
}

// Kernel to reorder elements based on sorted histogram
__global__ void reorderKernel(int *d_input, int *d_output, int *scan_hist, int exp, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < size) {
        int digit = (d_input[tid] / exp) % RADIX;
        int pos = atomicAdd(&scan_hist[digit], -1) - 1;  // Place in correct position
        d_output[pos] = d_input[tid];
    }
}

void radixSort(int *h_input, int size) {
    int *d_input, *d_output, *d_hist, *d_scan_hist;
    cudaCheckError(cudaMalloc(&d_input, size * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_output, size * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_hist, RADIX * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_scan_hist, RADIX * sizeof(int)));
    
    cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice));

    int maxVal = 0;
    for (int i = 0; i < size; i++) if (h_input[i] > maxVal) maxVal = h_input[i];

    for (int exp = 1; maxVal / exp > 0; exp *= RADIX) {
        cudaCheckError(cudaMemset(d_hist, 0, RADIX * sizeof(int)));

        int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        countSortKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_hist, exp, size);
        cudaDeviceSynchronize();

        scanKernel<<<1, RADIX>>>(d_hist, d_scan_hist);
        cudaDeviceSynchronize();

        reorderKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, d_scan_hist, exp, size);
        cudaDeviceSynchronize();

        cudaMemcpy(d_input, d_output, size * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaCheckError(cudaMemcpy(h_input, d_input, size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_scan_hist);
}

int main() {
    int h_input[N] = {329, 457, 657, 839, 436, 720, 355, 103, 278, 925, 699, 123, 234, 345, 456, 567};
    
    printf("Unsorted Array:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_input[i]);
    printf("\n");

    radixSort(h_input, N);

    printf("Sorted Array:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_input[i]);
    printf("\n");

    return 0;
}
