#include <cuda_runtime.h>
#include <stdio.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 16 
#define SHARED_MEM_WIDTH (TILE_WIDTH + MASK_WIDTH - 1) // Shared memory width with halo


// Kernel function for 2D conv with tiling with halo cells
__global__ void convolution2D(float *input, float *output, float *mask, int width, int height) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    __shared__ float sharedMem[SHARED_MEM_WIDTH][SHARED_MEM_WIDTH];

    int localRow = threadIdx.y + MASK_WIDTH / 2;
    int localCol = threadIdx.x + MASK_WIDTH / 2;
    
    if (row < height && col < width) {
        sharedMem[localRow][localCol] = input[row * width + col];
    }

    if (threadIdx.y < MASK_WIDTH / 2) {
        if (row - MASK_WIDTH / 2 >= 0) {
            sharedMem[threadIdx.y][localCol] = input[(row - MASK_WIDTH / 2) * width + col];
        }
        if (row + TILE_WIDTH < height) {
            sharedMem[threadIdx.y + TILE_WIDTH + MASK_WIDTH / 2][localCol] = input[(row + TILE_WIDTH) * width + col];
        }
    }

    if (threadIdx.x < MASK_WIDTH / 2) {
        if (col - MASK_WIDTH / 2 >= 0) {
            sharedMem[localRow][threadIdx.x] = input[row * width + (col - MASK_WIDTH / 2)];
        }
        if (col + TILE_WIDTH < width) {
            sharedMem[localRow][threadIdx.x + TILE_WIDTH + MASK_WIDTH / 2] = input[row * width + (col + TILE_WIDTH)];
        }
    }

    __syncthreads();


    //convolution for the pixel 
    if (row < height && col < width) {
        float value = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                int sharedRow = localRow + i - MASK_WIDTH / 2;
                int sharedCol = localCol + j - MASK_WIDTH / 2;
                value += sharedMem[sharedRow][sharedCol] * mask[i * MASK_WIDTH + j];
            }
        }
        output[row * width + col] = value;
    }
}

int main() {
    int width = 512;
    int height = 512;

    float *d_input, *d_output, *d_mask;
    float *h_input = (float *)malloc(width * height * sizeof(float));
    float *h_output = (float *)malloc(width * height * sizeof(float));


    //Simple blur mask
    float h_mask[MASK_WIDTH * MASK_WIDTH] = {1.0f / 9, 1.0f / 9, 1.0f / 9,
                                              1.0f / 9, 1.0f / 9, 1.0f / 9,
                                              1.0f / 9, 1.0f / 9, 1.0f / 9};

    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMalloc(&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    //input image random
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = (float)(i + j);
        }
    }

    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);

    convolution2D<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, width, height);

    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Input Image (first 5x5 pixels):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_input[i * width + j]);
        }
        printf("\n");
    }

    printf("\nOutput Image (after convolution, first 5x5 pixels):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_output[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
    free(h_input);
    free(h_output);

    return 0;
}
