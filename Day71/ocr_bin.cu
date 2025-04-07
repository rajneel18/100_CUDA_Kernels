#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define WIDTH  512
#define HEIGHT 512
#define THRESHOLD 128

__global__ void binarize_kernel(unsigned char* input, unsigned char* output, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

void generate_test_image(unsigned char* image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        image[i] = rand() % 256; // Random grayscale value
    }
}

int main() {
    int img_size = WIDTH * HEIGHT;
    unsigned char *h_input = (unsigned char*)malloc(img_size);
    unsigned char *h_output = (unsigned char*)malloc(img_size);

    generate_test_image(h_input, WIDTH, HEIGHT);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((WIDTH + threads.x - 1) / threads.x,
                (HEIGHT + threads.y - 1) / threads.y);

    binarize_kernel<<<blocks, threads>>>(d_input, d_output, WIDTH, HEIGHT, THRESHOLD);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    printf("First 10 binarized pixels:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
