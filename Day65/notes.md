# CUDA Implementation of Wasserstein Distance using Sinkhorn-Knopp Algorithm

## Overview
This CUDA C implementation computes the Wasserstein distance between two probability distributions using the Sinkhorn-Knopp algorithm for entropy regularized optimal transport.

## Key Components

### 1. **Compute Cost Matrix**
### 2. **Sinkhorn-Knopp Iteration**
### 3. **Compute Wasserstein Distance**


## Execution Flow
1. Allocate memory on host (`h_x, h_y`) and device (`d_x, d_y`).
2. Initialize input probability distributions with random values.
3. Copy data to GPU memory.
4. Compute cost matrix using `compute_cost_matrix` kernel.
5. Perform Sinkhorn-Knopp iterations using `sinkhorn_update`.
6. Compute the final Wasserstein distance using `compute_wasserstein`.
7. Copy result back to CPU and display.
8. Free allocated memory.
