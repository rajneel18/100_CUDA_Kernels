// fused_linear_softmax.cu
// Fused Linear Transformation + Softmax Kernel in CUDA

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void fused_linear_softmax_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* output,
    int input_dim,
    int output_dim
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_logits[];

    // Step 1: Matrix multiply (linear layer) + bias
    for (int j = tid; j < output_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            acc += input[batch_idx * input_dim + i] * weight[i * output_dim + j];
        }
        acc += bias[j];
        shared_logits[j] = acc;
    }

    __syncthreads();

    // Step 2: Find max logit for numerical stability
    float max_logit = -1e20f;
    for (int j = tid; j < output_dim; j += blockDim.x) {
        max_logit = fmaxf(max_logit, shared_logits[j]);
    }
    __shared__ float global_max;
    if (tid == 0) {
        global_max = -1e20f;
    }
    __syncthreads();

    atomicMax((int*)&global_max, __float_as_int(max_logit));
    __syncthreads();
    max_logit = __int_as_float(*(int*)&global_max);

    // Step 3: Exponentiate and compute sum
    float sum_exp = 0.0f;
    for (int j = tid; j < output_dim; j += blockDim.x) {
        shared_logits[j] = expf(shared_logits[j] - max_logit);
        sum_exp += shared_logits[j];
    }
    __shared__ float global_sum;
    if (tid == 0) global_sum = 0.0f;
    __syncthreads();

    atomicAdd(&global_sum, sum_exp);
    __syncthreads();

    // Step 4: Normalize logits to get softmax probabilities
    for (int j = tid; j < output_dim; j += blockDim.x) {
        output[batch_idx * output_dim + j] = shared_logits[j] / global_sum;
    }
}

void fused_linear_softmax(
    const float* input, const float* weight, const float* bias, float* output,
    int batch_size, int input_dim, int output_dim
) {
    int threads = 256;
    int shared_mem = output_dim * sizeof(float);
    fused_linear_softmax_kernel<<<batch_size, threads, shared_mem>>>(
        input, weight, bias, output, input_dim, output_dim
    );
}
int main() {
    const int batch_size = 2;
    const int input_dim = 3;
    const int output_dim = 4;

    float h_input[batch_size * input_dim] = {1, 2, 3, 4, 5, 6};
    float h_weight[input_dim * output_dim] = {0.1, 0.2, 0.3, 0.4,
                                              0.5, 0.6, 0.7, 0.8,
                                              0.9, 1.0, 1.1, 1.2};
    float h_bias[output_dim] = {0.01, 0.02, 0.03, 0.04};

    float* d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, batch_size * input_dim * sizeof(float));
    cudaMalloc(&d_weight, input_dim * output_dim * sizeof(float));
    cudaMalloc(&d_bias, output_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * output_dim * sizeof(float));

    cudaMemcpy(d_input, h_input, batch_size * input_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, input_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, output_dim * sizeof(float), cudaMemcpyHostToDevice);

    fused_linear_softmax(d_input, d_weight, d_bias, d_output, batch_size, input_dim, output_dim);

    float h_output[batch_size * output_dim];
    cudaMemcpy(h_output, d_output, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    for (int b = 0; b < batch_size; ++b) {
        printf("Batch %d:\n", b);
        for (int j = 0; j < output_dim; ++j) {
            printf("%f ", h_output[b * output_dim + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}