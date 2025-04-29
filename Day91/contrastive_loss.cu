#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

__device__ float euclidean_distance_squared(const float* a, const float* b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

__global__ void contrastive_loss_kernel(
    const float* x1, const float* x2, const int* labels,
    float* losses, int batch_size, int dim, float margin
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const float* a = x1 + idx * dim;
    const float* b = x2 + idx * dim;
    int label = labels[idx];

    float dist_sq = euclidean_distance_squared(a, b, dim);
    float loss;

    if (label == 1) {
        // Positive pair
        loss = dist_sq;
    } else {
        // Negative pair
        float dist = sqrtf(dist_sq);
        float diff = fmaxf(margin - dist, 0.0f);
        loss = diff * diff;
    }

    losses[idx] = 0.5f * loss;
}

int main() {
    const int batch_size = 4;
    const int dim = 3;
    const float margin = 2.0f;

    float h_x1[batch_size * dim] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        1.0, 1.0, 1.0,
        7.0, 8.0, 9.0
    };

    float h_x2[batch_size * dim] = {
        1.0, 2.0, 3.0,   // Same
        4.1, 5.1, 6.1,   // Slightly different
        3.0, 3.0, 3.0,   // Far
        7.0, 8.0, 9.0    // Same
    };

    int h_labels[batch_size] = {1, 1, 0, 0};

    float *d_x1, *d_x2, *d_losses;
    int *d_labels;
    float h_losses[batch_size];

    cudaMalloc((void**)&d_x1, batch_size * dim * sizeof(float));
    cudaMalloc((void**)&d_x2, batch_size * dim * sizeof(float));
    cudaMalloc((void**)&d_labels, batch_size * sizeof(int));
    cudaMalloc((void**)&d_losses, batch_size * sizeof(float));

    cudaMemcpy(d_x1, h_x1, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, batch_size * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    contrastive_loss_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x1, d_x2, d_labels, d_losses, batch_size, dim, margin);

    cudaMemcpy(h_losses, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Contrastive losses:\n");
    for (int i = 0; i < batch_size; ++i) {
        printf("Sample %d: Loss = %.6f\n", i, h_losses[i]);
    }

    cudaFree(d_x1);
    cudaFree(d_x2);
    cudaFree(d_labels);
    cudaFree(d_losses);

    return 0;
}
