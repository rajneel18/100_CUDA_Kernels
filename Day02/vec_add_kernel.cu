#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}

__global__ void vector_addition_kernel(int* d_A, int* d_B, int* d_C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

void vector_addition_cpu(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 10000000;  // Number of elements in the vector
    int size = N * sizeof(int);

    printf("Vector size: %d elements\n", N);

    // Allocate memory for vectors
    int* A = (int*)malloc(size);
    int* B = (int*)malloc(size);
    int* C = (int*)malloc(size);  // Result vector
    int* C_cpu = (int*)malloc(size);  // CPU result vector

    // Initialize vectors A and B with random values
    for (int i = 0; i < N; i++) {
        A[i] = rand() % 1000;
        B[i] = rand() % 1000;
    }

    int *d_A, *d_B, *d_C;

    dim3 block_size(256);
    dim3 grid_size(CEIL_DIV(N, block_size.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // GPU allocation
    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // Host to device transfer
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // Kernel execution
    cudaEventRecord(start);
    vector_addition_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // Device to host transfer
    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Running on CPU for comparison
    printf("\n>> Running vector addition on CPU...\n");
    clock_t ts = clock();
    vector_addition_cpu(A, B, C_cpu, N);
    clock_t te = clock();
    printf(">> Done\n");

    float elapsed_time = (te - ts) * 1000 / CLOCKS_PER_SEC;
    printf("Elapsed time (CPU): %.6f ms\n", elapsed_time);

    // Check if results match within an error tolerance (eps)
    bool match = true;
    float eps = 0.0001;
    for (int i = 0; i < N; i++) {
        if (fabs(C_cpu[i] - C[i]) > eps) {
            match = false;
            break;
        }
    }
    printf("\n>> Results match for CPU and GPU? ");
    printf("%s\n", match ? "true" : "false");

    // Cleanup
    free(A);
    free(B);
    free(C);
    free(C_cpu);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
