#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 32 // Grid size (N x N)
#define BLOCK_SIZE 16 // CUDA block size
#define ITERATIONS 100 // Number of game iterations

__global__ void gameOfLifeKernel(int *grid, int *newGrid, int n) {
    __shared__ int sharedGrid[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;

    if (x >= n || y >= n) return;

    // Load data into shared memory
    sharedGrid[localY][localX] = grid[y * n + x];

    // Load halo cells
    if (threadIdx.x == 0 && x > 0)
        sharedGrid[localY][0] = grid[y * n + (x - 1)];
    if (threadIdx.x == blockDim.x - 1 && x < n - 1)
        sharedGrid[localY][BLOCK_SIZE + 1] = grid[y * n + (x + 1)];
    if (threadIdx.y == 0 && y > 0)
        sharedGrid[0][localX] = grid[(y - 1) * n + x];
    if (threadIdx.y == blockDim.y - 1 && y < n - 1)
        sharedGrid[BLOCK_SIZE + 1][localX] = grid[(y + 1) * n + x];
    
    __syncthreads();

    // Compute number of live neighbors
    int liveNeighbors = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (!(i == 0 && j == 0)) {
                liveNeighbors += sharedGrid[localY + i][localX + j];
            }
        }
    }

    // Apply Game of Life rules
    int currentState = sharedGrid[localY][localX];
    if (currentState == 1 && (liveNeighbors < 2 || liveNeighbors > 3))
        newGrid[y * n + x] = 0;
    else if (currentState == 0 && liveNeighbors == 3)
        newGrid[y * n + x] = 1;
    else
        newGrid[y * n + x] = currentState;
}

void initializeGrid(int *grid, int n) {
    for (int i = 0; i < n * n; i++) {
        grid[i] = rand() % 2;
    }
}

void printGrid(int *grid, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%c ", grid[i * n + j] ? 'O' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int *h_grid, *h_newGrid;
    int *d_grid, *d_newGrid;
    size_t size = N * N * sizeof(int);

    // Allocate memory on host
    h_grid = (int *)malloc(size);
    h_newGrid = (int *)malloc(size);
    
    // Initialize grid
    initializeGrid(h_grid, N);
    printf("Initial Grid:\n");
    printGrid(h_grid, N);

    // Allocate memory on device
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_newGrid, size);
    
    // Copy initial grid to device
    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Run the game for multiple iterations
    for (int i = 0; i < ITERATIONS; i++) {
        gameOfLifeKernel<<<gridSize, blockSize>>>(d_grid, d_newGrid, N);
        cudaMemcpy(d_grid, d_newGrid, size, cudaMemcpyDeviceToDevice);
    }

    // Copy result back to host
    cudaMemcpy(h_newGrid, d_newGrid, size, cudaMemcpyDeviceToHost);
    
    printf("Final Grid:\n");
    printGrid(h_newGrid, N);

    // Free memory
    free(h_grid);
    free(h_newGrid);
    cudaFree(d_grid);
    cudaFree(d_newGrid);

    return 0;
}
