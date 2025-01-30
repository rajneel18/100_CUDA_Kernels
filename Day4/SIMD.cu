#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024  
#define THREADS_PER_BLOCK 256  


__global__ void squareArray(int *d_in, int *d_out, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];  
    }
}

int main() {
    int *h_in, *h_out;  
    int *d_in, *d_out; 

    size_t bytes = N * sizeof(int);

    h_in = (int*)malloc(bytes);
    h_out = (int*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;  // 1, 2, 3, ..., N
    }

    
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    squareArray<<<numBlocks, THREADS_PER_BLOCK>>>(d_in, d_out, N);

    
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    
    printf("First 10 Squared Values: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
