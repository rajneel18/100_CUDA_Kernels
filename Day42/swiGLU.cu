#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(error), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int rows_A, int cols_A, int cols_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_A && col < cols_B) {
        float sum = 0.0f;
        for (int k = 0; k < cols_A; k++) {
            sum += A[row * cols_A + k] * B[k * cols_B + col];
        }
        C[row * cols_B + col] = sum;
    }
}

// SwiGLU kernel
__global__ void swiglu_kernel(const float* XW, const float* XV, float* output,
                              int batch_size, int hidden_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < hidden_dim) {
        float xw = XW[row * hidden_dim + col];
        float xv = XV[row * hidden_dim + col];
        float swish = xw / (1.0f + expf(-xw)); // Swish(xW) = xW * sigmoid(xW)
        output[row * hidden_dim + col] = swish * xv; // SwiGLU = Swish(xW) * (xV)
    }
}

int main() {
    const int batch_size = 32;   
    const int input_dim = 512;    
    const int hidden_dim = 2048;  

    
    float *h_X, *h_W, *h_V, *h_XW, *h_XV, *h_output;
    h_X = (float*)malloc(batch_size * input_dim * sizeof(float));
    h_W = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    h_V = (float*)malloc(input_dim * hidden_dim * sizeof(float));
    h_XW = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    h_XV = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    h_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));

    if (h_X == NULL || h_W == NULL || h_V == NULL || 
        h_XW == NULL || h_XV == NULL || h_output == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < batch_size * input_dim; i++) {
        h_X[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < input_dim * hidden_dim; i++) {
        h_W[i] = (float)rand() / RAND_MAX - 0.5f; // Range [-0.5, 0.5]
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    float *d_X, *d_W, *d_V, *d_XW, *d_XV, *d_output;
    CUDA_CHECK(cudaMalloc(&d_X, batch_size * input_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, input_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, input_dim * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XW, batch_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_XV, batch_size * hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * hidden_dim * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16); // 256 threads per block
    dim3 blocksPerGrid(
        (hidden_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_W, d_XW, batch_size, input_dim, hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_V, d_XV, batch_size, input_dim, hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_XW, d_XV, d_output, batch_size, hidden_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, batch_size * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    printf("First few SwiGLU outputs:\n");
    for (int i = 0; i < 5 && i < batch_size; i++) {
        printf("%f ", h_output[i * hidden_dim]);
    }
    printf("\n");

    free(h_X);
    free(h_W);
    free(h_V);
    free(h_XW);
    free(h_XV);
    free(h_output);
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_XW));
    CUDA_CHECK(cudaFree(d_XV));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}