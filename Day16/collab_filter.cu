#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>

#define NUM_USERS 5
#define NUM_ITEMS 4
#define NUM_FACTORS 2     // Low-rank approximation dimension
#define LAMBDA 0.1f       // Regularization parameter
#define MAX_ITER 100      // Maximum iterations
#define LEARNING_RATE 0.01f  // Learning rate for gradient descent

// Error checking macro
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// Synthetic test matrix (user-item ratings)
__constant__ float R_const[NUM_USERS * NUM_ITEMS] = {
    5, 3, 0, 1,
    4, 0, 0, 1,
    1, 1, 0, 5,
    1, 0, 0, 4,
    0, 1, 5, 4
};

// CUDA kernel to update factor matrices P and Q
__global__ void update_factors(float* P, float* Q, int num_users, int num_items, int num_factors, float lambda, float alpha) {
    int user = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (user >= num_users) return;

    for (int i = 0; i < num_items; ++i) {
        float rating = R_const[user * num_items + i];
        
        if (rating > 0) {  // Only update for known ratings
            float prediction = 0.0f;
            
            // Calculate predicted rating
            for (int k = 0; k < num_factors; ++k) {
                prediction += P[user * num_factors + k] * Q[k * num_items + i];
            }
            
            float error = rating - prediction;
            
            // Update both P and Q matrices
            for (int k = 0; k < num_factors; ++k) {
                float p_old = P[user * num_factors + k];
                float q_old = Q[k * num_items + i];
                
                // Update using gradient descent
                atomicAdd(&P[user * num_factors + k], alpha * (error * q_old - lambda * p_old));
                atomicAdd(&Q[k * num_items + i], alpha * (error * p_old - lambda * q_old));
            }
        }
    }
}

// Host function to initialize matrices with random values
void initialize_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = 0.1f * ((float)rand() / RAND_MAX);
    }
}

// Calculate RMSE for convergence checking
float calculate_rmse(float* P, float* Q, float* R, int num_users, int num_items, int num_factors) {
    float rmse = 0.0f;
    int count = 0;
    
    for (int u = 0; u < num_users; ++u) {
        for (int i = 0; i < num_items; ++i) {
            if (R[u * num_items + i] > 0) {
                float pred = 0.0f;
                for (int k = 0; k < num_factors; ++k) {
                    pred += P[u * num_factors + k] * Q[k * num_items + i];
                }
                float diff = R[u * num_items + i] - pred;
                rmse += diff * diff;
                count++;
            }
        }
    }
    
    return sqrt(rmse / count);
}

int main() {
    // Host memory allocation
    float *h_R = new float[NUM_USERS * NUM_ITEMS];
    float *h_P = new float[NUM_USERS * NUM_FACTORS];
    float *h_Q = new float[NUM_FACTORS * NUM_ITEMS];
    
    // Initialize matrices
    for (int i = 0; i < NUM_USERS * NUM_ITEMS; ++i) {
        h_R[i] = R_const[i];
    }
    initialize_matrix(h_P, NUM_USERS, NUM_FACTORS);
    initialize_matrix(h_Q, NUM_FACTORS, NUM_ITEMS);
    
    // Device memory allocation
    float *d_P, *d_Q;
    cudaMalloc((void**)&d_P, NUM_USERS * NUM_FACTORS * sizeof(float));
    cudaMalloc((void**)&d_Q, NUM_FACTORS * NUM_ITEMS * sizeof(float));
    cudaCheckError();
    
    // Copy data to device
    cudaMemcpy(d_P, h_P, NUM_USERS * NUM_FACTORS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, NUM_FACTORS * NUM_ITEMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (NUM_USERS + blockSize - 1) / blockSize;
    
    std::cout << "Starting ALS optimization...\n";
    
    // Main optimization loop
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        update_factors<<<gridSize, blockSize>>>(d_P, d_Q, NUM_USERS, NUM_ITEMS, NUM_FACTORS, LAMBDA, LEARNING_RATE);
        cudaDeviceSynchronize();
        cudaCheckError();
        
        // Calculate and print RMSE every 10 iterations
        if (iter % 10 == 0) {
            cudaMemcpy(h_P, d_P, NUM_USERS * NUM_FACTORS * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Q, d_Q, NUM_FACTORS * NUM_ITEMS * sizeof(float), cudaMemcpyDeviceToHost);
            float rmse = calculate_rmse(h_P, h_Q, h_R, NUM_USERS, NUM_ITEMS, NUM_FACTORS);
            std::cout << "Iteration " << iter << ", RMSE: " << rmse << std::endl;
        }
    }
    
    // Copy results back to host
    cudaMemcpy(h_P, d_P, NUM_USERS * NUM_FACTORS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Q, d_Q, NUM_FACTORS * NUM_ITEMS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Print final matrices
    std::cout << "\nFinal User Factors (P):\n";
    for (int i = 0; i < NUM_USERS; ++i) {
        for (int j = 0; j < NUM_FACTORS; ++j) {
            std::cout << h_P[i * NUM_FACTORS + j] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nFinal Item Factors (Q):\n";
    for (int i = 0; i < NUM_FACTORS; ++i) {
        for (int j = 0; j < NUM_ITEMS; ++j) {
            std::cout << h_Q[i * NUM_ITEMS + j] << " ";
        }
        std::cout << "\n";
    }
    
    // Calculate final RMSE
    float final_rmse = calculate_rmse(h_P, h_Q, h_R, NUM_USERS, NUM_ITEMS, NUM_FACTORS);
    std::cout << "\nFinal RMSE: " << final_rmse << std::endl;
    
    // Cleanup
    cudaFree(d_P);
    cudaFree(d_Q);
    delete[] h_R;
    delete[] h_P;
    delete[] h_Q;
    
    return 0;
}