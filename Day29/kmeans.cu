#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define NUM_POINTS 1000
#define NUM_CLUSTERS 3
#define MAX_ITER 10
#define BLOCK_SIZE 256

//  Assign each point to the nearest centroid
__global__ void assignPointsToClusters(float *data, float *centroids, int *labels, int numPoints, int numClusters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numPoints) return;

    float minDist = FLT_MAX;
    int bestCluster = 0;
    
    float x = data[2 * idx];     
    float y = data[2 * idx + 1];  

    for (int c = 0; c < numClusters; c++) {
        float cx = centroids[2 * c];
        float cy = centroids[2 * c + 1];
        float dist = (x - cx) * (x - cx) + (y - cy) * (y - cy);

        if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
        }
    }
    labels[idx] = bestCluster;
}

// Update centroids using atomic operations
__global__ void updateCentroids(float *data, int *labels, float *centroids, int *clusterSizes, int numPoints, int numClusters) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numPoints) return;

    int cluster = labels[idx];
    atomicAdd(&centroids[2 * cluster], data[2 * idx]);     
    atomicAdd(&centroids[2 * cluster + 1], data[2 * idx + 1]); 
    atomicAdd(&clusterSizes[cluster], 1); 
}

// Compute final centroids
__global__ void finalizeCentroids(float *centroids, int *clusterSizes, int numClusters) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (c >= numClusters) return;

    if (clusterSizes[c] > 0) {
        centroids[2 * c] /= clusterSizes[c];
        centroids[2 * c + 1] /= clusterSizes[c];
    }
}

void initializeData(float *data, int numPoints) {
    for (int i = 0; i < numPoints; i++) {
        data[2 * i] = rand() % 100;   // Random X
        data[2 * i + 1] = rand() % 100; // Random Y
    }
}

void initializeCentroids(float *centroids, int numClusters) {
    for (int i = 0; i < numClusters; i++) {
        centroids[2 * i] = rand() % 100;
        centroids[2 * i + 1] = rand() % 100;
    }
}

int main() {
    float *h_data, *h_centroids;
    int *h_labels;
    float *d_data, *d_centroids;
    int *d_labels, *d_clusterSizes;

    h_data = (float*)malloc(NUM_POINTS * 2 * sizeof(float));
    h_centroids = (float*)malloc(NUM_CLUSTERS * 2 * sizeof(float));
    h_labels = (int*)malloc(NUM_POINTS * sizeof(int));

    initializeData(h_data, NUM_POINTS);
    initializeCentroids(h_centroids, NUM_CLUSTERS);

    cudaMalloc(&d_data, NUM_POINTS * 2 * sizeof(float));
    cudaMalloc(&d_centroids, NUM_CLUSTERS * 2 * sizeof(float));
    cudaMalloc(&d_labels, NUM_POINTS * sizeof(int));
    cudaMalloc(&d_clusterSizes, NUM_CLUSTERS * sizeof(int));

    cudaMemcpy(d_data, h_data, NUM_POINTS * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, NUM_CLUSTERS * 2 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSizePoints((NUM_POINTS + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 gridSizeClusters((NUM_CLUSTERS + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        cudaMemset(d_clusterSizes, 0, NUM_CLUSTERS * sizeof(int));

        assignPointsToClusters<<<gridSizePoints, blockSize>>>(d_data, d_centroids, d_labels, NUM_POINTS, NUM_CLUSTERS);
        cudaDeviceSynchronize();

        cudaMemset(d_centroids, 0, NUM_CLUSTERS * 2 * sizeof(float));

        updateCentroids<<<gridSizePoints, blockSize>>>(d_data, d_labels, d_centroids, d_clusterSizes, NUM_POINTS, NUM_CLUSTERS);
        cudaDeviceSynchronize();

        finalizeCentroids<<<gridSizeClusters, blockSize>>>(d_centroids, d_clusterSizes, NUM_CLUSTERS);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_labels, d_labels, NUM_POINTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, NUM_CLUSTERS * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final Centroids:\n");
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Cluster %d: (%.2f, %.2f)\n", i, h_centroids[2 * i], h_centroids[2 * i + 1]);
    }

    
    free(h_data);
    free(h_centroids);
    free(h_labels);
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_clusterSizes);

    return 0;
}
