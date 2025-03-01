#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// KComputes logits and BCE loss in one step
__global__ void binaryCrossEntropyFusedKernel(float *y_true, float *X, float *W, float *loss, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float y = y_true[idx];

    // Compute logit
    float logit = 0.0f;
    for (int d = 0; d < D; d++) {
        logit += X[idx * D + d] * W[d];  // Matrix-vector multiplication
    }

    // Sigmoid activation 
    float y_hat = 1.0f / (1.0f + expf(-logit));

    // Avoid log(0) issues
    y_hat = fmaxf(fminf(y_hat, 1.0f - 1e-7), 1e-7);

    // Compute BCE loss
    float sample_loss = -(y * logf(y_hat) + (1 - y) * logf(1 - y_hat));

    // Accumulate loss using atomicAdd (to avoid race conditions)
    atomicAdd(loss, sample_loss);
}

// Function to run BCE loss computation on CUDA with chunk processing
float binaryCrossEntropyFusedCUDA(float *y_true, float *X, float *W, int N, int D, int chunk_size) {
    float *d_y_true, *d_X, *d_W, *d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_y_true, N * sizeof(float));
    cudaMalloc(&d_X, chunk_size * D * sizeof(float));  
    cudaMalloc(&d_W, D * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

   
    cudaMemcpy(d_y_true, y_true, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int num_chunks = (N + chunk_size - 1) / chunk_size;  

    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int chunk_N = min(chunk_size, N - offset);

        cudaMemcpy(d_X, X + offset * D, chunk_N * D * sizeof(float), cudaMemcpyHostToDevice);

        int blocksPerGrid = (chunk_N + threadsPerBlock - 1) / threadsPerBlock;
        binaryCrossEntropyFusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_y_true + offset, d_X, d_W, d_loss, chunk_N, D);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_y_true);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_loss);

    return h_loss / N;
}

int main() {
    int N = 1000;  
    int D = 10;    
    int chunk_size = 256;

    // Allocate host memory
    float *y_true = (float*)malloc(N * sizeof(float));
    float *X = (float*)malloc(N * D * sizeof(float));
    float *W = (float*)malloc(D * sizeof(float));

    for (int i = 0; i < N; i++) {
        y_true[i] = rand() % 2;  
        for (int d = 0; d < D; d++) {
            X[i * D + d] = static_cast<float>(rand()) / RAND_MAX;  
        }
    }
    for (int d = 0; d < D; d++) {
        W[d] = static_cast<float>(rand()) / RAND_MAX;  
    }

    float loss = binaryCrossEntropyFusedCUDA(y_true, X, W, N, D, chunk_size);

    
    printf("Binary Cross-Entropy Loss: %f\n", loss);

    free(y_true);
    free(X);
    free(W);

    return 0;
}