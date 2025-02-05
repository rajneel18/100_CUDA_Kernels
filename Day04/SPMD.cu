#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

__global__ void ele_wise_subtract(int* d_A, int* d_B, int* d_C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        d_C[idx] = d_A[idx] - d_B[idx];
    }
}

int main() {
    int N = 1024;
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    A = (int*)malloc(N*sizeof(int));
    B = (int*)malloc(N*sizeof(int));
    C = (int*)malloc(N*sizeof(int));

    for(int i = 0; i<N; i++){
        A[i] = i;
        B[i] = i;
    }
    
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N* sizeof(int));
    cudaMalloc(&d_C, N* sizeof(int));

    cudaMemcpy(d_A, A, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N* sizeof(int), cudaMemcpyHostToDevice);
    

    int blockSize = 256;
    int gridSize = (N+ blockSize-1)/blockSize;
    
    ele_wise_subtract<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %d\n", i, C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;

    }