#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_USERS 943          
#define NUM_ITEMS 1682         
#define NUM_FACTORS 10         

// CUDA kernel for predicting ratings
__global__ void predict_ratings(float *P, float *Q, float *user_bias, float *item_bias, float *mean_user_ratings, float *predictions) {
    int user = blockIdx.x * blockDim.x + threadIdx.x; // Thread ID represents a user
    int item = blockIdx.y * blockDim.y + threadIdx.y; // Thread ID represents an item

    if (user < NUM_USERS && item < NUM_ITEMS) {
        float prediction = mean_user_ratings[user] + user_bias[user] + item_bias[item];
        for (int k = 0; k < NUM_FACTORS; k++) {
            prediction += P[user * NUM_FACTORS + k] * Q[item * NUM_FACTORS + k];
        }
        predictions[user * NUM_ITEMS + item] = prediction;
    }
}

void load_data(const char *filename, float *array, int size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        exit(-1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &array[i]) != 1) {
            printf("Error reading data at index %d\n", i);
            exit(-1);
        }
    }
    fclose(file);
}


// Helper function to measure kernel execution time
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

int main() {
    // Load mean user ratings and initialize data
    float *mean_user_ratings = (float *)malloc(NUM_USERS * sizeof(float));
    load_data("mean_user_ratings.txt", mean_user_ratings, NUM_USERS);

    // Simulate trained P, Q, user_bias, and item_bias
    float *P = (float *)malloc(NUM_USERS * NUM_FACTORS * sizeof(float));
    float *Q = (float *)malloc(NUM_ITEMS * NUM_FACTORS * sizeof(float));
    float *user_bias = (float *)malloc(NUM_USERS * sizeof(float));
    float *item_bias = (float *)malloc(NUM_ITEMS * sizeof(float));
    for (int i = 0; i < NUM_USERS * NUM_FACTORS; i++) P[i] = 0.1f * ((float)rand() / RAND_MAX);
    for (int i = 0; i < NUM_ITEMS * NUM_FACTORS; i++) Q[i] = 0.1f * ((float)rand() / RAND_MAX);
    for (int i = 0; i < NUM_USERS; i++) user_bias[i] = 0.1f * ((float)rand() / RAND_MAX);
    for (int i = 0; i < NUM_ITEMS; i++) item_bias[i] = 0.1f * ((float)rand() / RAND_MAX);

    // Allocate device memory
    float *d_P, *d_Q, *d_user_bias, *d_item_bias, *d_mean_user_ratings, *d_predictions;
    checkCudaError(cudaMalloc((void **)&d_P, NUM_USERS * NUM_FACTORS * sizeof(float)), "Malloc d_P");
    checkCudaError(cudaMalloc((void **)&d_Q, NUM_ITEMS * NUM_FACTORS * sizeof(float)), "Malloc d_Q");
    checkCudaError(cudaMalloc((void **)&d_user_bias, NUM_USERS * sizeof(float)), "Malloc d_user_bias");
    checkCudaError(cudaMalloc((void **)&d_item_bias, NUM_ITEMS * sizeof(float)), "Malloc d_item_bias");
    checkCudaError(cudaMalloc((void **)&d_mean_user_ratings, NUM_USERS * sizeof(float)), "Malloc d_mean_user_ratings");
    checkCudaError(cudaMalloc((void **)&d_predictions, NUM_USERS * NUM_ITEMS * sizeof(float)), "Malloc d_predictions");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_P, P, NUM_USERS * NUM_FACTORS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy d_P");
    checkCudaError(cudaMemcpy(d_Q, Q, NUM_ITEMS * NUM_FACTORS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy d_Q");
    checkCudaError(cudaMemcpy(d_user_bias, user_bias, NUM_USERS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy d_user_bias");
    checkCudaError(cudaMemcpy(d_item_bias, item_bias, NUM_ITEMS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy d_item_bias");
    checkCudaError(cudaMemcpy(d_mean_user_ratings, mean_user_ratings, NUM_USERS * sizeof(float), cudaMemcpyHostToDevice), "Memcpy d_mean_user_ratings");

    // Launch kernel with optimized grid and block size
    dim3 blockSize(32, 32); // 32x32 threads per block
    dim3 gridSize((NUM_USERS + 31) / 32, (NUM_ITEMS + 31) / 32);

    // Measure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    predict_ratings<<<gridSize, blockSize>>>(d_P, d_Q, d_user_bias, d_item_bias, d_mean_user_ratings, d_predictions);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.4f ms\n", milliseconds);

    // Copy predictions back to host
    float *predictions = (float *)malloc(NUM_USERS * NUM_ITEMS * sizeof(float));
    checkCudaError(cudaMemcpy(predictions, d_predictions, NUM_USERS * NUM_ITEMS * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy predictions");

    // Calculate RMSE
    FILE *matrix_file = fopen("normalized_matrix.txt", "r");
    if (!matrix_file) {
        printf("Error loading normalized matrix.\n");
        return -1;
    }

    float rmse = 0;
    int count = 0;
    for (int user = 0; user < NUM_USERS; user++) {
        for (int item = 0; item < NUM_ITEMS; item++) {
            float actual;
            fscanf(matrix_file, "%f", &actual);
            if (actual != 0) {  // Only evaluate where actual ratings exist
                float error = actual - predictions[user * NUM_ITEMS + item];
                rmse += error * error;
                count++;
            }
        }
    }
    fclose(matrix_file);
    rmse = sqrt(rmse / count);
    printf("RMSE: %.4f\n", rmse);

    // Free device memory
    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_user_bias);
    cudaFree(d_item_bias);
    cudaFree(d_mean_user_ratings);
    cudaFree(d_predictions);

    // Free host memory
    free(mean_user_ratings);
    free(P);
    free(Q);
    free(user_bias);
    free(item_bias);
    free(predictions);

    return 0;
}
