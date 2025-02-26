#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gaussianBlur1DRow(float* d_out, const float* d_in, int width, int height, 
                                  const float* d_kernel, int kernel_size) {
    __shared__ float sharedMem[BLOCK_SIZE + 10];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    int half_k = kernel_size / 2;

    if (x >= width || y >= height) return;

    sharedMem[threadIdx.x + half_k] = d_in[y * width + x];

    if (threadIdx.x < half_k) {
        sharedMem[threadIdx.x] = d_in[y * width + max(x - half_k, 0)];
        sharedMem[threadIdx.x + blockDim.x + half_k] = d_in[y * width + min(x + half_k, width - 1)];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = -half_k; k <= half_k; k++) {
        sum += sharedMem[threadIdx.x + k + half_k] * d_kernel[k + half_k];
    }
    
    d_out[y * width + x] = sum;
}

__global__ void gaussianBlur1DCol(float* d_out, const float* d_in, int width, int height, 
                                  const float* d_kernel, int kernel_size) {
    __shared__ float sharedMem[BLOCK_SIZE + 10];

    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_k = kernel_size / 2;

    if (x >= width || y >= height) return;

    sharedMem[threadIdx.y + half_k] = d_in[y * width + x];

    if (threadIdx.y < half_k) {
        sharedMem[threadIdx.y] = d_in[max(y - half_k, 0) * width + x];
        sharedMem[threadIdx.y + blockDim.y + half_k] = d_in[min(y + half_k, height - 1) * width + x];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int k = -half_k; k <= half_k; k++) {
        sum += sharedMem[threadIdx.y + k + half_k] * d_kernel[k + half_k];
    }
    
    d_out[y * width + x] = sum;
}

void gaussianKernel(float* kernel, int size, float sigma) {
    float sum = 0.0f;
    int half = size / 2;
    for (int i = -half; i <= half; i++) {
        kernel[i + half] = expf(-0.5f * (i * i) / (sigma * sigma));
        sum += kernel[i + half];
    }
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    int width = 5, height = 5;
    float h_in[25] = {10, 20, 30, 40, 50,
                      60, 70, 80, 90, 100,
                      110, 120, 130, 140, 150,
                      160, 170, 180, 190, 200,
                      210, 220, 230, 240, 250};

    float *d_in, *d_out, *d_kernel;
    int kernel_size = 3;
    float sigma = 1.0f;
    float h_kernel[3];

    cudaMalloc(&d_in, width * height * sizeof(float));
    cudaMalloc(&d_out, width * height * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));

    gaussianKernel(h_kernel, kernel_size, sigma);
    cudaMemcpy(d_in, h_in, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    gaussianBlur1DRow<<<gridSize, blockSize>>>(d_out, d_in, width, height, d_kernel, kernel_size);
    cudaMemcpy(h_in, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Blurred Image:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", h_in[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
    return 0;
}
