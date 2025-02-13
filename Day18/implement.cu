#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_USERS 3
#define NUM_ITEMS 3
#define NUM_FACTORS 2  // Low-rank approximation dimension
#define LAMBDA 0.1     // Regularization parameter
#define MAX_ITER 100   // Maximum iterations
#define LEARNING_RATE 0.01  // Learning rate

// Synthetic test matrix (user-item ratings)
float R[NUM_USERS][NUM_ITEMS] = {
    {5, 3, 0},
    {4, 0, 1},
    {1, 1, 5}
};

// CUDA kernel to update user factors (P) and biases (bu)
__global__ void update_user_factors(float *R, float *P, float *Q, float *bu, float *bi, int num_users, int num_items, int num_factors, float lambda, float alpha) {
    int user = blockIdx.x * blockDim.x + threadIdx.x;
    if (user >= num_users) return;

    for (int i = 0; i < num_items; ++i) {
        if (R[user * num_items + i] > 0) {  // Only update for known ratings
            float prediction = bu[user] + bi[i];
            for (int k = 0; k < num_factors; ++k) {
                prediction += P[user * num_factors + k] * Q[k * num_items + i];
            }
            float error = R[user * num_items + i] - prediction;

            for (int k = 0; k < num_factors; ++k) {
                P[user * num_factors + k] += alpha * (error * Q[k * num_items + i] - lambda * P[user * num_factors + k]);
            }
            bu[user] += alpha * (error - lambda * bu[user]);
        }
    }
}

// CUDA kernel to update item factors (Q) and biases (bi)
__global__ void update_item_factors(float *R, float *P, float *Q, float *bu, float *bi, int num_users, int num_items, int num_factors, float lambda, float alpha) {
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= num_items) return;

    for (int u = 0; u < num_users; ++u) {
        if (R[u * num_items + item] > 0) {  // Only update for known ratings
            float prediction = bu[u] + bi[item];
            for (int k = 0; k < num_factors; ++k) {
                prediction += P[u * num_factors + k] * Q[k * num_items + item];
            }
            float error = R[u * num_items + item] - prediction;

            for (int k = 0; k < num_factors; ++k) {
                Q[k * num_items + item] += alpha * (error * P[u * num_factors + k] - lambda * Q[k * num_items + item]);
            }
            bi[item] += alpha * (error - lambda * bi[item]);
        }
    }
}

// Function to initialize matrices with small random values
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = 0.1f * ((float)rand() / RAND_MAX);
    }
}

// Function to calculate the predicted rating
float predict_rating(float *P, float *Q, float *bu, float *bi, int user, int item, int num_factors) {
    float prediction = bu[user] + bi[item];
    for (int k = 0; k < num_factors; ++k) {
        prediction += P[user * num_factors + k] * Q[k * NUM_ITEMS + item];
    }
    return prediction;
}

// Function to compute RMSE
float compute_rmse(float *R, float *P, float *Q, float *bu, float *bi, int num_users, int num_items, int num_factors) {
    float error = 0.0f;
    int count = 0;
    for (int u = 0; u < num_users; ++u) {
        for (int i = 0; i < num_items; ++i) {
            if (R[u * num_items + i] > 0) {
                float prediction = predict_rating(P, Q, bu, bi, u, i, num_factors);
                error += pow(R[u * num_items + i] - prediction, 2);
                count++;
            }
        }
    }
    return sqrt(error / count);
}

// Host function to run ALS
void runALS(float *R, int num_users, int num_items, int num_factors) {
    float *P, *Q, *bu, *bi, *d_R;
    int size_R = num_users * num_items * sizeof(float);
    int size_P = num_users * num_factors * sizeof(float);
    int size_Q = num_factors * num_items * sizeof(float);
    int size_bu = num_users * sizeof(float);
    int size_bi = num_items * sizeof(float);

    // Allocate host memory
    float *h_P = (float *)malloc(size_P);
    float *h_Q = (float *)malloc(size_Q);
    float *h_bu = (float *)malloc(size_bu);
    float *h_bi = (float *)malloc(size_bi);

    // Initialize P, Q, bu, and bi with random values
    initialize_matrix(h_P, num_users, num_factors);
    initialize_matrix(h_Q, num_factors, num_items);
    for (int i = 0; i < num_users; ++i) h_bu[i] = 0.0f;
    for (int i = 0; i < num_items; ++i) h_bi[i] = 0.0f;

    // Allocate device memory
    cudaMalloc((void **)&d_R, size_R);
    cudaMalloc((void **)&P, size_P);
    cudaMalloc((void **)&Q, size_Q);
    cudaMalloc((void **)&bu, size_bu);
    cudaMalloc((void **)&bi, size_bi);

    // Copy data to device
    cudaMemcpy(d_R, R, size_R, cudaMemcpyHostToDevice);
    cudaMemcpy(P, h_P, size_P, cudaMemcpyHostToDevice);
    cudaMemcpy(Q, h_Q, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(bu, h_bu, size_bu, cudaMemcpyHostToDevice);
    cudaMemcpy(bi, h_bi, size_bi, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSizeUsers = (num_users + blockSize - 1) / blockSize;
    int gridSizeItems = (num_items + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        update_user_factors<<<gridSizeUsers, blockSize>>>(d_R, P, Q, bu, bi, num_users, num_items, num_factors, LAMBDA, LEARNING_RATE);
        cudaDeviceSynchronize();
        update_item_factors<<<gridSizeItems, blockSize>>>(d_R, P, Q, bu, bi, num_users, num_items, num_factors, LAMBDA, LEARNING_RATE);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(h_P, P, size_P, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q, Q, size_Q, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bu, bu, size_bu, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bi, bi, size_bi, cudaMemcpyDeviceToHost);

    // Compute RMSE
    float rmse = compute_rmse(&R[0][0], h_P, h_Q, h_bu, h_bi, num_users, num_items, num_factors);
    printf("RMSE: %.4f\n", rmse);

    // Free device memory
    cudaFree(d_R);
    cudaFree(P);
    cudaFree(Q);
    cudaFree(bu);
    cudaFree(bi);

    // Free host memory
    free(h_P);
    free(h_Q);
    free(h_bu);
    free(h_bi);
}

int main() {
    printf("Running ALS with Biases on CUDA with a synthetic test matrix...\n");
    runALS(&R[0][0], NUM_USERS, NUM_ITEMS, NUM_FACTORS);
    return 0;
}
