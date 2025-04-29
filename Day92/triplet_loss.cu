#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float euclidean_distance_squared(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

__global__ void triplet_loss_kernel(
    const float* anchor, const float* positive, const float* negative,
    float* losses, int batch_size, int dim, float margin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* a = anchor + idx * dim;
    const float* p = positive + idx * dim;
    const float* n = negative + idx * dim;

    float d_ap = euclidean_distance_squared(a, p, dim);
    float d_an = euclidean_distance_squared(a, n, dim);
    float loss = fmaxf(d_ap - d_an + margin, 0.0f);
    losses[idx] = loss;
}

int main() {
    const int batch_size = 3;
    const int dim = 4;
    const float margin = 0.5f;

    float h_anchor[batch_size * dim] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        1.0, 0.0, 1.0, 0.0
    };

    float h_positive[batch_size * dim] = {
        1.1, 2.1, 3.1, 4.1,
        5.2, 6.2, 7.2, 8.2,
        0.9, 0.1, 0.9, 0.1
    };

    float h_negative[batch_size * dim] = {
        2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0,
        3.0, 2.0, 3.0, 2.0
    };

    float h_losses[batch_size] = {0};

    float *d_anchor, *d_positive, *d_negative, *d_losses;
    size_t vec_bytes = batch_size * dim * sizeof(float);
    size_t loss_bytes = batch_size * sizeof(float);

    cudaMalloc(&d_anchor, vec_bytes);
    cudaMalloc(&d_positive, vec_bytes);
    cudaMalloc(&d_negative, vec_bytes);
    cudaMalloc(&d_losses, loss_bytes);

    cudaMemcpy(d_anchor, h_anchor, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, h_positive, vec_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, h_negative, vec_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    triplet_loss_kernel<<<blocks, threads>>>(d_anchor, d_positive, d_negative, d_losses, batch_size, dim, margin);
    cudaDeviceSynchronize();

    cudaMemcpy(h_losses, d_losses, loss_bytes, cudaMemcpyDeviceToHost);

    printf("Triplet Losses:\n");
    for (int i = 0; i < batch_size; ++i) {
        printf("Sample %d: %.6f\n", i, h_losses[i]);
    }

    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_losses);

    return 0;
}
