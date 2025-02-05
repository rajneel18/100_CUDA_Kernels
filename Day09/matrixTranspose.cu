#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16  // Tile size for shared memory optimization

__global__ void transposeShared(float *d_out, float *d_in, int width, int height) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // Avoid bank conflicts by padding
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = d_in[y * width + x];
    }
    __syncthreads();
    
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < height && y < width) {
        d_out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void matrixTranspose(float *h_out, float *h_in, int width, int height) {
    float *d_in, *d_out;
    size_t size = width * height * sizeof(float);
    
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);
    
    transposeShared<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    const int width = 32, height = 16;
    float h_in[width * height], h_out[width * height];
    
    for (int i = 0; i < width * height; i++) {
        h_in[i] = static_cast<float>(i);
    }
    
    matrixTranspose(h_out, h_in, width, height);
    
    printf("Transposed Matrix:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            printf("%.1f ", h_out[i * height + j]);
        }
        printf("\n");
    }
    
    return 0;
}  
