#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax(float* input, float* output, int N) {
    int idx = threadIdx.x;
    if (idx >= N) return;

    // Compute exponentials
    float exp_val = expf(input[idx]);

    // Compute sum of exponentials using shared memory
    __shared__ float sum_exp;
    if (idx == 0) sum_exp = 0.0f;
    __syncthreads();

    atomicAdd(&sum_exp, exp_val);
    __syncthreads();

    // Compute softmax
    output[idx] = exp_val / sum_exp;
}

void softmaxHost(float* input, float* output, int N) {
    float *d_input, *d_output;
    size_t size = N * sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording time
    cudaEventRecord(start);
    softmax<<<1, N>>>(d_input, d_output, N);
    cudaEventRecord(stop);

    // Wait for completion
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

    // Print execution time
    printf("CUDA Softmax Execution Time: %.6f ms\n", milliseconds);
}

int main() {
    int N = 5;
    float h_input[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_output[5];

    softmaxHost(h_input, h_output, N);

    printf("Softmax Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%f\n", h_output[i]);
    }
    return 0;
}
