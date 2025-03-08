#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024  
#define BLOCK_SIZE 256
#define ITERATIONS 1000
#define EPSILON 1e-6f  // Small value to prevent log(0)

// Gradient Descent Update
__global__ void gradient_descent(float *x, float *grad, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] -= alpha * grad[i];
    }
}

// Mirror Descent Update (Entropy-based)
__global__ void mirror_descent(float *x, float *grad, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float new_val = logf(fmaxf(x[i], EPSILON)) - alpha * grad[i];  // Ensure positivity
        x[i] = expf(new_val);
    }
}

// Compute Gradient (Ax - b)
__global__ void compute_gradient(float *A, float *x, float *b, float *grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        grad[i] = sum - b[i];
    }
}

// initialize A, b, and x
void initialize_data(float *A, float *b, float *x, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = (float)(rand() % 10);  // Random values for b
        x[i] = 1.0f;  // Initialize x to positive values for Mirror Descent
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (i == j) ? 2.0f : 0.5f;  // Diagonal-dominant matrix
        }
    }
}

// Loss f(x) = 0.5 * x^T A x - b^T x
float compute_loss(float *A, float *x, float *b, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float Ax = 0.0f;
        for (int j = 0; j < n; j++) {
            Ax += A[i * n + j] * x[j];
        }
        loss += 0.5f * Ax * x[i] - b[i] * x[i];
    }
    return loss;
}

int main() {
    float *h_A, *h_b, *h_x, *h_grad, *h_x_md;
    float *d_x, *d_x_md, *d_grad, *d_A, *d_b;
    float alpha = 0.001f;

    h_A = (float*)malloc(N * N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_x = (float*)malloc(N * sizeof(float));
    h_x_md = (float*)malloc(N * sizeof(float)); // Separate array for Mirror Descent
    h_grad = (float*)malloc(N * sizeof(float));

    // Initialize A, b, and x
    initialize_data(h_A, h_b, h_x, N);
    memcpy(h_x_md, h_x, N * sizeof(float)); // Copy x for Mirror Descent

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_x_md, N * sizeof(float));
    cudaMalloc(&d_grad, N * sizeof(float));
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_md, h_x_md, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    float elapsedTime;

    // Gradient Descent
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        compute_gradient<<<gridSize, blockSize>>>(d_A, d_x, d_b, d_grad, N);
        gradient_descent<<<gridSize, blockSize>>>(d_x, d_grad, alpha, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Gradient Descent Time: %f ms\n", elapsedTime);

    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    float loss_gd = compute_loss(h_A, h_x, h_b, N);
    printf("Gradient Descent Final Loss: %f\n", loss_gd);

    // Mirror Descent 
    cudaEventRecord(start, 0);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        compute_gradient<<<gridSize, blockSize>>>(d_A, d_x_md, d_b, d_grad, N);
        mirror_descent<<<gridSize, blockSize>>>(d_x_md, d_grad, alpha, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Mirror Descent Time: %f ms\n", elapsedTime);

    cudaMemcpy(h_x_md, d_x_md, N * sizeof(float), cudaMemcpyDeviceToHost);
    float loss_md = compute_loss(h_A, h_x_md, h_b, N);
    printf("Mirror Descent Final Loss: %f\n", loss_md);

    if (loss_gd < loss_md) {
        printf("Gradient Descent converged better.\n");
    } else {
        printf("Mirror Descent converged better.\n");
    }

    free(h_A);
    free(h_b);
    free(h_x);
    free(h_x_md);
    free(h_grad);
    cudaFree(d_x);
    cudaFree(d_x_md);
    cudaFree(d_grad);
    cudaFree(d_A);
    cudaFree(d_b);

    return 0;
}
