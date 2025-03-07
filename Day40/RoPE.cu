#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// CUDA kernel to apply RoPE
__global__ void rope_kernel(
    float* queries, float* keys,
    int batch_size, int seq_len, int num_heads, int head_dim, float base) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_tokens = batch_size * seq_len * num_heads;
    
    if (idx >= total_tokens) return;

    int batch_idx = idx / (seq_len * num_heads);
    int seq_idx = (idx / num_heads) % seq_len;
    int head_idx = idx % num_heads;

    int base_idx = batch_idx * seq_len * num_heads * head_dim +
                   seq_idx * num_heads * head_dim +
                   head_idx * head_dim;

    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(base, (float)(i) / head_dim);
        float theta = seq_idx * freq;

        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float q_real = queries[base_idx + i];
        float q_img = queries[base_idx + i + 1];

        queries[base_idx + i] = q_real * cos_theta - q_img * sin_theta;
        queries[base_idx + i + 1] = q_real * sin_theta + q_img * cos_theta;

        float k_real = keys[base_idx + i];
        float k_img = keys[base_idx + i + 1];

        keys[base_idx + i] = k_real * cos_theta - k_img * sin_theta;
        keys[base_idx + i + 1] = k_real * sin_theta + k_img * cos_theta;
    }
}

// Host function to launch RoPE kernel
void apply_rope(
    float* d_queries, float* d_keys,
    int batch_size, int seq_len, int num_heads, int head_dim, float base) {

    int total_tokens = batch_size * seq_len * num_heads;
    int block_size = 256;
    int grid_size = (total_tokens + block_size - 1) / block_size;

    rope_kernel<<<grid_size, block_size>>>(
        d_queries, d_keys, batch_size, seq_len, num_heads, head_dim, base);
}

int main() {
    int batch_size = 1;
    int seq_len = 8;
    int num_heads = 4;
    int head_dim = 16;
    float base = 10000.0f;

    int total_size = batch_size * seq_len * num_heads * head_dim;
    size_t mem_size = total_size * sizeof(float);

    float* h_queries = (float*)malloc(mem_size);
    float* h_keys = (float*)malloc(mem_size);

    for (int i = 0; i < total_size; i++) {
        h_queries[i] = (float)(i % 10) / 10.0f;
        h_keys[i] = (float)((i + 1) % 10) / 10.0f;
    }

    float *d_queries, *d_keys;
    cudaMalloc((void**)&d_queries, mem_size);
    cudaMalloc((void**)&d_keys, mem_size);

    cudaMemcpy(d_queries, h_queries, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, h_keys, mem_size, cudaMemcpyHostToDevice);

    apply_rope(d_queries, d_keys, batch_size, seq_len, num_heads, head_dim, base);

    cudaMemcpy(h_queries, d_queries, mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_keys, d_keys, mem_size, cudaMemcpyDeviceToHost);

    printf("Modified Queries:\n");
    for (int i = 0; i < total_size; i++) {
        printf("%.4f ", h_queries[i]);
        if ((i + 1) % head_dim == 0) printf("\n");
    }
    printf("\n");

    printf("Modified Keys:\n");
    for (int i = 0; i < total_size; i++) {
        printf("%.4f ", h_keys[i]);
        if ((i + 1) % head_dim == 0) printf("\n");
    }
    printf("\n");

    free(h_queries);
    free(h_keys);
    cudaFree(d_queries);
    cudaFree(d_keys);

    return 0;
}
