#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

__global__ void rgbToGreyscale(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        unsigned char r = d_input[idx];
        unsigned char g = d_input[idx + 1];
        unsigned char b = d_input[idx + 2];
        d_output[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

void convertToGreyscaleCUDA(const unsigned char* h_input, unsigned char* h_output, int width, int height) {
    int imageSize = width * height;
    int rgbSize = imageSize * 3;

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, rgbSize);
    cudaMalloc(&d_output, imageSize);

    cudaMemcpy(d_input, h_input, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    rgbToGreyscale<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    string inputImagePath = "input_image.jpg";
    string outputImagePath = "output_cuda_image.jpg";

    Mat image = imread(inputImagePath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not open the image file." << endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    unsigned char* h_input = image.data;
    unsigned char* h_output = new unsigned char[width * height];

    // Measure start time
    auto start = high_resolution_clock::now();

    convertToGreyscaleCUDA(h_input, h_output, width, height);

    // Measure end time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "CUDA processing time: " << duration.count() << " milliseconds." << endl;

    Mat outputImage(height, width, CV_8UC1, h_output);
    imwrite(outputImagePath, outputImage);

    delete[] h_output;

    return 0;
}
