#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 256 

using namespace cv;

    // CUDA kernel to compute the histogram
    __global__ void computeHistogram(const unsigned char *input, int *hist, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;

        atomicAdd(&hist[input[idx]], 1);  // Increment the corresponding bin in the histogram
    }

    // CUDA kernel to compute the CDF and normalize it
    __global__ void computeCDFAndNormalize(int *hist, int *cdf, int totalPixels) {
        int idx = threadIdx.x;
        __shared__ int tempHist[256];  // Shared memory for the histogram

        if (idx < 256) tempHist[idx] = hist[idx];
        __syncthreads();

        if (idx == 0) {
            int sum = 0;
            for (int i = 0; i < 256; i++) {
                sum += tempHist[i];
                cdf[i] = (float)(sum - tempHist[0]) / (totalPixels - tempHist[0]) * 255.0f;
                if (cdf[i] < 0) cdf[i] = 0;
                if (cdf[i] > 255) cdf[i] = 255;
            }
        }
    }

    // Equilization to the histogram
    __global__ void applyEqualization(const unsigned char *input, unsigned char *output, int *cdf, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= width * height) return;

        output[idx] = cdf[input[idx]];  
    }

int main() {
    
    Mat inputImage = imread("input.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        printf("Error: Could not load image.\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    
    Mat outputImage(height, width, CV_8UC1);
    int histogram[256] = {0}, cdf[256] = {0};

    
    unsigned char *d_input, *d_output;
    int *d_hist, *d_cdf;
    cudaMalloc((void **)&d_input, imageSize);
    cudaMalloc((void **)&d_output, imageSize);
    cudaMalloc((void **)&d_hist, 256 * sizeof(int));
    cudaMalloc((void **)&d_cdf, 256 * sizeof(int));

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));  // Initialize the histogram on the device

    dim3 block(BLOCK_SIZE);
    dim3 grid((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Compute the histogram
    computeHistogram<<<grid, block>>>(d_input, d_hist, width, height);
    cudaDeviceSynchronize();

    // Compute the CDF and normalize it
    computeCDFAndNormalize<<<1, 256>>>(d_hist, d_cdf, width * height);
    cudaDeviceSynchronize();

    // Apply histogram equalization to the image
    applyEqualization<<<grid, block>>>(d_input, d_output, d_cdf, width, height);
    cudaDeviceSynchronize();

    
    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    
    imwrite("output.jpg", outputImage);
    printf("Histogram Equalization complete. Saved as output.jpg.\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hist);
    cudaFree(d_cdf);

    return 0;
}
