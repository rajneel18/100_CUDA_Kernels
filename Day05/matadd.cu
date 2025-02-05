#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void matrixAdd(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < n && col < n) {
   int idx = row * n + col;
   C[idx] = A[idx] + B[idx];
  }
}

int main() {
 int size = N * N * sizeof(float);
 float *h_A, *h_B, *h_C;
 float *d_A, *d_B, *d_C;

 h_A = (float *) malloc(size);
 h_B = (float *) malloc(size);
 h_C = (float *) malloc(size);

 if (h_A == NULL || h_B == NULL || h_C == NULL) {
   printf("Memory allocation failed on host\n");
   return -1;
 }

 for(int i = 0; i < N * N; i++) {
  h_A[i] = 1037.0f;
  h_B[i] = 6575.0f;
 }

 cudaMalloc(&d_A, size);
 cudaMalloc(&d_B, size);
 cudaMalloc(&d_C, size);

 if (d_A == NULL || d_B == NULL || d_C == NULL) {
   printf("Memory allocation failed on device\n");
   return -1;
 }

 cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
 cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

 dim3 blockDim(16, 16);
 dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

 matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

 cudaError_t err = cudaGetLastError();
 if (err != cudaSuccess) {
     printf("CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
     return -1;
 }

 cudaDeviceSynchronize();

 cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

 printf("Resulting matrix (first 5x5 elements):\n");
 for(int i = 0; i < 5; i++) {
  for(int j = 0; j < 5; j++) {
   printf("%f ", h_C[i * N + j]);
  }
  printf("\n");
 }

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);
 free(h_A);
 free(h_B);
 free(h_C);

 return 0;
}
