#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024
#define MAX_ITER 1000

// Kernel to compute Mandelbrot set
__global__ void mandelbrotKernel(unsigned char *image, int width, int height, float x_min, float x_max, float y_min, float y_max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float real = x_min + (x / (float)width) * (x_max - x_min);
    float imag = y_min + (y / (float)height) * (y_max - y_min);
    
    float zx = 0.0, zy = 0.0;
    int iteration = 0;

    while (zx * zx + zy * zy < 4.0f && iteration < MAX_ITER) {
        float temp = zx * zx - zy * zy + real;
        zy = 2.0 * zx * zy + imag;
        zx = temp;
        iteration++;
    }

    int pixel_index = (y * width + x) * 3;
    unsigned char color = (unsigned char)(255 * iteration / MAX_ITER);
    
    image[pixel_index] = color;      // Red
    image[pixel_index + 1] = color;  // Green
    image[pixel_index + 2] = color;  // Blue
}

// Save image as PPM
void savePPM(const char *filename, unsigned char *image, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(image, 1, width * height * 3, fp);
    fclose(fp);
}

int main() {
    unsigned char *h_image, *d_image;
    
    size_t imageSize = WIDTH * HEIGHT * 3 * sizeof(unsigned char);
    h_image = (unsigned char*)malloc(imageSize);
    cudaMalloc((void**)&d_image, imageSize);

    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    mandelbrotKernel<<<gridDim, blockDim>>>(d_image, WIDTH, HEIGHT, -2.0f, 1.0f, -1.5f, 1.5f);
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);

    savePPM("mandelbrot.ppm", h_image, WIDTH, HEIGHT);

    free(h_image);
    cudaFree(d_image);

    return 0;
}
