#include <stdio.h>
#include <cuda_runtime.h>

#define BATCH 2
#define DIM 8

// CUDA kernel to perform residual connection: output = input + scale * residual
__global__ void residual_add(float* input, float* residual, float* output, float scale, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        int batch_id = blockIdx.y;
        int offset = batch_id * dim + idx;

        output[offset] = input[offset] + scale * residual[offset];
    }
}

int main() {
    int size = BATCH * DIM * sizeof(float);

    float h_input[BATCH * DIM] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        2, 4, 6, 8, 10, 12, 14, 16
    };
    float h_residual[BATCH * DIM] = {
        0.5, 1, 0.5, 1, 0.5, 1, 0.5, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    };
    float h_output[BATCH * DIM];

    float *d_input, *d_residual, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_residual, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual, size, cudaMemcpyHostToDevice);

    dim3 threads(8);
    dim3 blocks(1, BATCH);
    float scale = 0.1f;

    residual_add<<<blocks, threads>>>(d_input, d_residual, d_output, scale, DIM);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Residual Output:\n");
    for (int b = 0; b < BATCH; b++) {
        for (int i = 0; i < DIM; i++) {
            printf("%.2f ", h_output[b * DIM + i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_residual);
    cudaFree(d_output);

    return 0;
}
