#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define M 10
#define BLOCK_SIZE 256

__global__ void dotProductKernel(float *a, float *b, float *result, int n) {
    __shared__ float cache[BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0.0f;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }

    if (cacheIndex == 0) {
        atomicAdd(result, cache[0]);
    }
}

__global__ void vectorUpdateKernel(float *x, float *d, float alpha, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        x[tid] += alpha * d[tid];
    }
}

__global__ void computeGradientKernel(float *g, float *Qx, float *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        g[tid] = Qx[tid] - b[tid];
    }
}

void lbfgs(float *d_x, float *d_g, float *d_Q, float *d_b, int n, int max_iters) {
    float *s[M], *y[M];
    float alpha[M], rho[M];

    for (int i = 0; i < M; i++) {
        cudaMalloc(&s[i], n * sizeof(float));
        cudaMalloc(&y[i], n * sizeof(float));
    }

    float *d_Qx, *d_direction, *d_temp, *d_dot_result;
    cudaMalloc(&d_Qx, n * sizeof(float));
    cudaMalloc(&d_direction, n * sizeof(float));
    cudaMalloc(&d_temp, n * sizeof(float));
    cudaMalloc(&d_dot_result, sizeof(float));

    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    for (int iter = 0; iter < max_iters; iter++) {
        cudaMemcpy(d_Qx, d_x, n * sizeof(float), cudaMemcpyDeviceToDevice);
        computeGradientKernel<<<grid, block>>>(d_g, d_Qx, d_b, n);

        int m = iter < M ? iter : M;
        float beta;
        
        for (int i = m - 1; i >= 0; i--) {
            dotProductKernel<<<grid, block>>>(s[i], d_g, d_dot_result, n);
            cudaMemcpy(&alpha[i], d_dot_result, sizeof(float), cudaMemcpyDeviceToHost);
            alpha[i] *= rho[i];
        }

        cudaMemcpy(d_direction, d_g, n * sizeof(float), cudaMemcpyDeviceToDevice);

        for (int i = 0; i < m; i++) {
            dotProductKernel<<<grid, block>>>(y[i], d_direction, d_dot_result, n);
            cudaMemcpy(&beta, d_dot_result, sizeof(float), cudaMemcpyDeviceToHost);
            beta *= rho[i];

            float scale = alpha[i] - beta;
            vectorUpdateKernel<<<grid, block>>>(d_direction, s[i], scale, n);
        }

        float step_size = 0.01;
        vectorUpdateKernel<<<grid, block>>>(d_x, d_direction, -step_size, n);

        cudaMemcpy(s[iter % M], d_x, n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(y[iter % M], d_g, n * sizeof(float), cudaMemcpyDeviceToDevice);
        dotProductKernel<<<grid, block>>>(y[iter % M], s[iter % M], d_dot_result, n);
        cudaMemcpy(&rho[iter % M], d_dot_result, sizeof(float), cudaMemcpyDeviceToHost);
        rho[iter % M] = 1.0f / rho[iter % M];
    }

    for (int i = 0; i < M; i++) {
        cudaFree(s[i]);
        cudaFree(y[i]);
    }
    cudaFree(d_Qx);
    cudaFree(d_direction);
    cudaFree(d_temp);
    cudaFree(d_dot_result);
}

int main() {
    float *h_x, *h_g, *h_Q, *h_b;
    float *d_x, *d_g, *d_Q, *d_b;

    h_x = (float*)malloc(N * sizeof(float));
    h_g = (float*)malloc(N * sizeof(float));
    h_Q = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_x[i] = 0.0f;
        h_b[i] = 1.0f;
        h_Q[i] = 2.0f;
    }

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_g, N * sizeof(float));
    cudaMalloc(&d_Q, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    lbfgs(d_x, d_g, d_Q, d_b, N, 100);

    cudaMemcpy(h_x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Optimized solution (first 10 values): \n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_x[i]);
    }
    printf("\n");

    free(h_x); free(h_g); free(h_Q); free(h_b);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_Q); cudaFree(d_b);

    return 0;
}
