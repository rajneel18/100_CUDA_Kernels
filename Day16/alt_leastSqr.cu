#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_USERS 5
#define NUM_ITEMS 4
#define NUM_FACTORS 2  // Low-rank approximation dimension
#define LAMBDA 0.1     // Regularization parameter
#define MAX_ITER 100   // Maximum iterations
#define LEARNING_RATE 0.01  // Learning rate for ALS

// Synthetic test matrix (user-item ratings)
float R[NUM_USERS][NUM_ITEMS] = {
    {5, 3, 0, 1},
    {4, 0, 0, 1},
    {1, 1, 0, 5},
    {1, 0, 0, 4},
    {0, 1, 5, 4}
};

// CUDA kernel to update factor matrices P and Q
__global__ void update_factors(float *R, float *P, float *Q, int num_users, int num_items, int num_factors, float lambda, float alpha) {
    int user = blockIdx.x * blockDim.x + threadIdx.x;
    if (user >= num_users) return;

    for (int i = 0; i < num_items; ++i) {
        if (R[user * num_items + i] > 0) {  // Only update for known ratings
            float prediction = 0.0f;
            for (int k = 0; k < num_factors; ++k) {
                prediction += P[user * num_factors + k] * Q[k * num_items + i];
            }
            float error = R[user * num_items + i] - prediction;

            for (int k = 0; k < num_factors; ++k) {
                float p_old = P[user * num_factors + k];
                float q_old = Q[k * num_items + i];
                P[user * num_factors + k] += alpha * (error * q_old - lambda * p_old);
                Q[k * num_items + i] += alpha * (error * p_old - lambda * q_old);
            }
        }
    }
}

// Function to initialize factor matrices with random values
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = 0.1f * ((float)rand() / RAND_MAX);  // Small random values
    }
}

// Host function to run ALS
void runALS(float *R, int num_users, int num_items, int num_factors) {
    float *P, *Q, *d_R;
    int size_R = num_users * num_items * sizeof(float);
    int size_P = num_users * num_factors * sizeof(float);
    int size_Q = num_factors * num_items * sizeof(float);

    // Allocate host memory for factor matrices
    float *h_P = (float *)malloc(size_P);
    float *h_Q = (float *)malloc(size_Q);

    // Initialize P and Q with random values
    initialize_matrix(h_P, num_users, num_factors);
    initialize_matrix(h_Q, num_factors, num_items);

    // Allocate device memory
    cudaMalloc((void **)&d_R, size_R);
    cudaMalloc((void **)&P, size_P);
    cudaMalloc((void **)&Q, size_Q);

    // Copy data to device
    cudaMemcpy(d_R, R, size_R, cudaMemcpyHostToDevice);
    cudaMemcpy(P, h_P, size_P, cudaMemcpyHostToDevice);
    cudaMemcpy(Q, h_Q, size_Q, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (num_users + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        update_factors<<<gridSize, blockSize>>>(d_R, P, Q, num_users, num_items, num_factors, LAMBDA, LEARNING_RATE);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(h_P, P, size_P, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q, Q, size_Q, cudaMemcpyDeviceToHost);

    // Print the factor matrices
    printf("Matrix P (User factors):\n");
    for (int i = 0; i < num_users; ++i) {
        for (int j = 0; j < num_factors; ++j) {
            printf("%.4f ", h_P[i * num_factors + j]);
        }
        printf("\n");
    }

    printf("\nMatrix Q (Item factors):\n");
    for (int i = 0; i < num_factors; ++i) {
        for (int j = 0; j < num_items; ++j) {
            printf("%.4f ", h_Q[i * num_items + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_R);
    cudaFree(P);
    cudaFree(Q);

    // Free host memory
    free(h_P);
    free(h_Q);
}

int main() {
    printf("Running ALS on CUDA with a test matrix...\n");
    runALS(&R[0][0], NUM_USERS, NUM_ITEMS, NUM_FACTORS);
    return 0;
}
