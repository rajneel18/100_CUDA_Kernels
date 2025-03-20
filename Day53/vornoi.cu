#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define WIDTH 1024
#define HEIGHT 1024
#define NUM_SEEDS 50

struct Seed {
    float x, y;
    int color;
};

// Kernel to compute Voronoi cells
__global__ void computeVoronoi(int *output, Seed *seeds, int num_seeds, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int closestSeed = 0;
    float minDist = 1e10;

    for (int i = 0; i < num_seeds; i++) {
        float dx = x - seeds[i].x;
        float dy = y - seeds[i].y;
        float dist = dx * dx + dy * dy;

        if (dist < minDist) {
            minDist = dist;
            closestSeed = i;
        }
    }

    output[y * width + x] = seeds[closestSeed].color; // Assign color
}

// Kernel to initialize random seed points
__global__ void initSeeds(Seed *seeds, curandState *randStates, int num_seeds, int width, int height) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_seeds) return;

    curand_init(idx, 0, 0, &randStates[idx]);
    seeds[idx].x = curand_uniform(&randStates[idx]) * width;
    seeds[idx].y = curand_uniform(&randStates[idx]) * height;
    seeds[idx].color = curand(&randStates[idx]) % 0xFFFFFF;  // Random color
}

int main() {
    int *d_output, *h_output;
    Seed *d_seeds;
    curandState *d_randStates;

    cudaMalloc(&d_output, WIDTH * HEIGHT * sizeof(int));
    cudaMalloc(&d_seeds, NUM_SEEDS * sizeof(Seed));
    cudaMalloc(&d_randStates, NUM_SEEDS * sizeof(curandState));

    h_output = (int*)malloc(WIDTH * HEIGHT * sizeof(int));

    // Initialize random seed points
    initSeeds<<<1, NUM_SEEDS>>>(d_seeds, d_randStates, NUM_SEEDS, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Compute Voronoi diagram
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);
    computeVoronoi<<<numBlocks, threadsPerBlock>>>(d_output, d_seeds, NUM_SEEDS, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Save as PPM image (basic output format)
    FILE *fp = fopen("voronoi.ppm", "w");
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        int color = h_output[i];
        fprintf(fp, "%d %d %d ", (color >> 16) & 255, (color >> 8) & 255, color & 255);
    }
    fclose(fp);

    printf("Voronoi diagram saved as 'voronoi.ppm'.\n");

    // Free memory
    cudaFree(d_output);
    cudaFree(d_seeds);
    cudaFree(d_randStates);
    free(h_output);

    return 0;
}
