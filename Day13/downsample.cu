#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

using namespace cv;

__global__ void downsampleAndQuantize(unsigned char *input, unsigned char *output, int width, int height, int newWidth, int newHeight, int quantizationLevel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < newWidth && y < newHeight) {
        int origX = x * (width / newWidth);
        int origY = y * (height / newHeight);
        int origIndex = origY * width + origX;

        unsigned char pixel = input[origIndex];
        output[y * newWidth + x] = (pixel / quantizationLevel) * quantizationLevel;
    }
}

int main() {
    Mat inputImage = imread("input_image.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int newWidth = width / 2;
    int newHeight = height / 2;
    int quantizationLevel = 16;

    size_t originalSize = width * height * sizeof(unsigned char);
    size_t downsampledSize = newWidth * newHeight * sizeof(unsigned char);

    Mat outputImage(newHeight, newWidth, CV_8UC1);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, originalSize);
    cudaMalloc((void **)&d_output, downsampledSize);

    cudaMemcpy(d_input, inputImage.data, originalSize, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((newWidth + BLOCK_SIZE - 1) / BLOCK_SIZE, (newHeight + BLOCK_SIZE - 1) / BLOCK_SIZE);

    downsampleAndQuantize<<<grid, block>>>(d_input, d_output, width, height, newWidth, newHeight, quantizationLevel);

    cudaMemcpy(outputImage.data, d_output, downsampledSize, cudaMemcpyDeviceToHost);

    imwrite("output.jpg", outputImage);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
