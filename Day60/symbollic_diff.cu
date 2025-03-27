#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 256

// CUDA Kernel to compute symbolic derivative of ax^n
__global__ void symbolic_diff(double *a, double *n, double *da, double *dn) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    da[idx] = a[idx] * n[idx];   // Coefficient derivative
    dn[idx] = n[idx] - 1;        // Power derivative
}

int main() {
    int N = 5; 
    
    double h_a[5] = {3, 2, 5, 4, 1};  // Coefficients of terms (e.g., 3x^4, 2x^3, ...)
    double h_n[5] = {4, 3, 2, 1, 0};  // Powers of x
    double h_da[5], h_dn[5];          // Results of differentiation
    
    double *d_a, *d_n, *d_da, *d_dn;
    
    cudaMalloc((void **)&d_a, N * sizeof(double));
    cudaMalloc((void **)&d_n, N * sizeof(double));
    cudaMalloc((void **)&d_da, N * sizeof(double));
    cudaMalloc((void **)&d_dn, N * sizeof(double));

    cudaMemcpy(d_a, h_a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, N * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    symbolic_diff<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_n, d_da, d_dn);
    
    cudaMemcpy(h_da, d_da, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dn, d_dn, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Symbolic Derivative:\n");
    for (int i = 0; i < N; i++) {
        printf("d(%.1fx^%.1f)/dx = %.1fx^%.1f\n", h_a[i], h_n[i], h_da[i], h_dn[i]);
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_n);
    cudaFree(d_da);
    cudaFree(d_dn);
    
    return 0;
}
