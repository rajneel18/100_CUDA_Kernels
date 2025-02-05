#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 7       
#define MASK_WIDTH 5  

// CUDA Kernel for 1D Convolution
__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    float Pvalue = 0.0f;
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {  
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    if (i < Width) {
        P[i] = Pvalue;  
    }
}

int main() {
    float h_N[WIDTH] = {1, 2, 3, 4, 5, 6, 7};          
    float h_M[MASK_WIDTH] = {0.2, 0.5, 0.2, -0.2, 0.1}; 
    float h_P[WIDTH] = {0};                            

    float *d_N, *d_M, *d_P;
    
    cudaMalloc((void**)&d_N, WIDTH * sizeof(float));
    cudaMalloc((void**)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, WIDTH * sizeof(float));

    cudaMemcpy(d_N, h_N, WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (WIDTH + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1D_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_N, d_M, d_P, MASK_WIDTH, WIDTH);

    cudaMemcpy(h_P, d_P, WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Processed Output (P): ");
    for (int i = 0; i < WIDTH; i++)
        printf("%.2f ", h_P[i]);
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
