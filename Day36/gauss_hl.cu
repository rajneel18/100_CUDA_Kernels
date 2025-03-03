#include <stdio.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include <math.h>

#define BLOCK_SIZE 256
#define SIGMA 5.0f  // Standard deviation for Gaussian kernel

using namespace cv;

// CUDA kernel to compute histogram
__global__ void computeHistogram(const unsigned char *input, int *hist, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    atomicAdd(&hist[input[idx]], 1);
}

// CUDA kernel to apply Gaussian smoothing on histogram
__global__ void smoothHistogram(const int *hist, float *probHist, int totalPixels) {
    int idx = threadIdx.x;
    __shared__ float tempHist[256];

    if (idx < 256) tempHist[idx] = hist[idx];
    __syncthreads();

    if (idx < 256) {
        float sum = 0.0f;
        float weightSum = 0.0f;

        // Apply Gaussian kernel
        for (int j = -3 * SIGMA; j <= 3 * SIGMA; j++) {
            int neighbor = idx + j;
            if (neighbor >= 0 && neighbor < 256) {
                float weight = expf(-0.5f * (j * j) / (SIGMA * SIGMA)) / (SIGMA * sqrtf(2.0f * M_PI));
                sum += tempHist[neighbor] * weight;
                weightSum += weight;
            }
        }

        probHist[idx] = sum / (weightSum * totalPixels);  // Normalize to get probability
    }
}

int main() {
    Mat inputImage = imread("input_image.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        printf("Error: Could not load image.\n");
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    int histogram[256] = {0};
    float probabilityHist[256] = {0};

    unsigned char *d_input;
    int *d_hist;
    float *d_probHist;

    cudaMalloc((void **)&d_input, imageSize);
    cudaMalloc((void **)&d_hist, 256 * sizeof(int));
    cudaMalloc((void **)&d_probHist, 256 * sizeof(float));

    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    dim3 block(BLOCK_SIZE);
    dim3 grid((width * height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    computeHistogram<<<grid, block>>>(d_input, d_hist, width, height);
    cudaDeviceSynchronize();

    smoothHistogram<<<1, 256>>>(d_hist, d_probHist, width * height);
    cudaDeviceSynchronize();

    cudaMemcpy(probabilityHist, d_probHist, 256 * sizeof(float), cudaMemcpyDeviceToHost);

    // Print probability distribution
    printf("Histogram Probability Distribution:\n");
    for (int i = 0; i < 256; i++) {
        printf("Bin %d: %f\n", i, probabilityHist[i]);
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaFree(d_probHist);

    return 0;
}
