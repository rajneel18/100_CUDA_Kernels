#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 5  
int main() {
    
    float h_x[N] = {1, 2, 3, 4, 5};
    float h_y[N] = {10, 20, 30, 40, 50};
    float alpha = 2.0f; 

    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // y = alpha * x + y
    cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant y vector: ");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
