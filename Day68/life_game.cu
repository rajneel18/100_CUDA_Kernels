#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 1024
#define NUM_NEURONS 10

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

__global__ void life_step_kernel(const bool* current, bool* next, int width) {
    __shared__ bool shared_grid[BLOCK_SIZE+2][BLOCK_SIZE+2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Load data into shared memory including halo cells
    if (x < width && y < width) {
        // Center cell
        shared_grid[ty+1][tx+1] = current[y * width + x];
        
        // Edges
        if (tx == 0 && x > 0)
            shared_grid[ty+1][0] = current[y * width + (x-1)];
        if (tx == BLOCK_SIZE-1 && x < width-1)
            shared_grid[ty+1][BLOCK_SIZE+1] = current[y * width + (x+1)];
        if (ty == 0 && y > 0)
            shared_grid[0][tx+1] = current[(y-1) * width + x];
        if (ty == BLOCK_SIZE-1 && y < width-1)
            shared_grid[BLOCK_SIZE+1][tx+1] = current[(y+1) * width + x];
    }
    
    __syncthreads();
    
    if (x < width && y < width) {
        int neighbors = 0;
        
        // Count neighbors
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int ny = ty + 1 + dy;
                int nx = tx + 1 + dx;
                if (shared_grid[ny][nx]) neighbors++;
            }
        }
        
        // Apply Life rules
        bool current_cell = shared_grid[ty+1][tx+1];
        bool alive = (current_cell && (neighbors == 2 || neighbors == 3)) ||
                    (!current_cell && neighbors == 3);
        
        next[y * width + x] = alive;
    }
}

void print_grid(const bool* grid, int width, int neuron) {
    printf("Neuron %d: ", neuron);
    for (int i = 0; i < width; i++) {
        printf("%d", grid[i] ? 1 : 0);
    }
    printf("\n");
}

int main() {
    bool *h_grid, *h_next;
    bool *d_grid, *d_next;
    
    // Allocate host memory
    h_grid = (bool*)calloc(GRID_SIZE * NUM_NEURONS, sizeof(bool));
    h_next = (bool*)calloc(GRID_SIZE * NUM_NEURONS, sizeof(bool));
    
    // Initialize with random pattern
    for (int n = 0; n < NUM_NEURONS; n++) {
        for (int i = 0; i < GRID_SIZE; i++) {
            h_grid[n * GRID_SIZE + i] = (rand() % 100 < 15);  // 15% chance of being alive
        }
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_grid, GRID_SIZE * NUM_NEURONS * sizeof(bool)));
    CHECK_CUDA(cudaMalloc(&d_next, GRID_SIZE * NUM_NEURONS * sizeof(bool)));
    
    // Copy initial state to device
    CHECK_CUDA(cudaMemcpy(d_grid, h_grid, GRID_SIZE * NUM_NEURONS * sizeof(bool), cudaMemcpyHostToDevice));
    
    // Setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (GRID_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Main simulation loop
    for (int n = 0; n < NUM_NEURONS; n++) {
        life_step_kernel<<<blocks, threads>>>(d_grid + n * GRID_SIZE, 
                                            d_next + n * GRID_SIZE, 
                                            GRID_SIZE);
    }
    
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_next, d_next, GRID_SIZE * NUM_NEURONS * sizeof(bool), cudaMemcpyDeviceToHost));
    
    // Print results
    for (int n = 0; n < NUM_NEURONS; n++) {
        print_grid(h_next + n * GRID_SIZE, GRID_SIZE, n);
    }
    
    // Cleanup
    cudaFree(d_grid);
    cudaFree(d_next);
    free(h_grid);
    free(h_next);
    
    return 0;
}
