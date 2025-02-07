#include <stdio.h>
#include <cuda_runtime.h>

__global__ void parallelPrefixSum(int *d_array, int n) {
    int tid = threadIdx.x;
    
    // Up sweep phase
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            d_array[index] += d_array[index - stride];
        }
        __syncthreads();
    }

    // prepare for down-sweep
    if (tid == 0) d_array[n - 1] = 0;
    __syncthreads();

    // Down sweep phase 
    for (int stride = n / 2; stride >= 1; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            int temp = d_array[index - stride];
            d_array[index - stride] = d_array[index];
            d_array[index] += temp;
        }
        __syncthreads();
    }
}

int main() {
    int h_array[] = {12, 23, 32, 48, 56, 64, 70, 88};  // Example array
    int n = sizeof(h_array) / sizeof(h_array[0]);

    int *d_array;
    cudaMalloc((void**)&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    parallelPrefixSum<<<1, n / 2>>>(d_array, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Prefix sum: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    cudaFree(d_array);

    return 0;
}
