# Convolution with Tiling with Halo cells

## Introduction

Convolution is a fundamental operation in image processing and neural networks, where a kernel (filter) is applied to an image (or matrix) to compute outputs such as blurring, sharpening, edge detection, etc. When implementing convolution on GPUs, two key optimization strategies are commonly used: **tiling** and **halo**.

### Tiling
Tiling refers to the process of dividing the input image or matrix into smaller blocks (tiles) that can be processed by individual threads or thread blocks in parallel. This helps to efficiently utilize the parallel processing power of GPUs. 

In tiling, each block of threads works on a submatrix (tile) of the input, performing the convolution operation on the elements of that tile.

### Halo cells
A **halo** is a border around each tile that is needed to ensure that each thread in the tile can compute the convolution correctly, including elements near the edges of the tile. These border elements from neighboring tiles are stored in shared memory, allowing threads to access data outside their local tile. This is essential for correct convolution at the boundaries.

The combination of tiling and halo ensures that:
- Threads work on small, manageable chunks (tiles) of data.
- Threads can access neighboring data (halo) to handle convolution near the borders without needing global memory access every time.

## Key Concepts

### 1. **Tiling for Efficient Computation**
   - The image is divided into **tiles** (smaller blocks).
   - Each tile is processed independently by a block of threads.
   - The size of the tile is typically chosen based on the GPU's architecture and the available shared memory.
   - Tiling ensures better memory locality, as threads in the block access neighboring elements within the tile.

### 2. **Halo for Handling Boundaries**
   - The **halo** refers to the boundary elements that surround the tile.
   - These boundary elements are needed to compute convolution correctly at the edges of the tile.
   - Threads use **shared memory** to store both the tile's data and the halo data, allowing efficient access without global memory access.

### 3. **Shared Memory**
   - Shared memory is faster than global memory and is accessible by all threads within the same block.
   - In tiled convolution, the shared memory is used to store both the local tile data and the halo data.
   - Synchronization (`__syncthreads()`) is needed to ensure that all threads have finished loading data into shared memory before performing the convolution.

## Algorithm Steps

1. **Divide the image into tiles**: 
   - The image is split into tiles of size `(TILE_WIDTH x TILE_WIDTH)`.
   - Each thread block is assigned a tile to work on.

2. **Load the tile into shared memory**:
   - Threads load data from global memory into shared memory.
   - The data loaded includes not only the elements in the tile but also the halo elements (data from neighboring tiles).

3. **Handle boundaries using halo**:
   - Threads load halo elements for the rows and columns near the borders of the tile.
   - This ensures that the threads can access neighboring data when computing the convolution for boundary pixels.

4. **Perform the convolution**:
   - Each thread computes the convolution for one pixel in the tile.
   - It accesses the corresponding data in the shared memory (including the halo data).
   - The result of the convolution is stored in the output image.

5. **Synchronize threads**: 
   - Before performing the convolution, threads synchronize to ensure all data is loaded into shared memory.

6. **Store the result**:
   - After computing the convolution, the output is written back to global memory.

## Code Walkthrough

### Kernel Function

The kernel function for 2D convolution with tiling and halo typically follows these steps:

1. **Determine the global pixel indices** (`row`, `col`) for each thread.
2. **Declare shared memory** to store the local tile and halo.
3. **Load data into shared memory**:
   - Each thread loads data from global memory into the shared memory.
   - Load halo elements for boundary pixels (i.e., elements outside the current tile but needed for convolution).
4. **Synchronize threads** to ensure all data is loaded into shared memory.
5. **Perform the convolution** by iterating through the mask and applying it to the data in shared memory.
6. **Write the result** to the output image.

