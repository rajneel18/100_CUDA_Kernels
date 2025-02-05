# Thread Scheduling in CUDA

Thread scheduling in CUDA refers to how threads are assigned to the GPU's processing units (Streaming Multiprocessors, or SMs) for execution. Efficient thread scheduling is crucial for achieving high performance and maximizing resource utilization on the GPU.

## Thread Hierarchy:
- **Threads:** The smallest unit of execution in CUDA. Each thread performs a specific task.
- **Blocks:** A group of threads that are executed together. Each block runs on a single SM.
- **Grids:** A collection of blocks that can run in parallel on the GPU. The grid size defines how many blocks can be launched.

## Types of Thread Scheduling:
- **Static Scheduling:** Thread blocks are mapped to SMs in a fixed manner. This scheduling is efficient for workloads where the number of blocks is known ahead of time.
- **Dynamic Scheduling:** Threads are scheduled dynamically at runtime. This allows for more flexibility but may result in less efficient execution compared to static scheduling.

## Optimizing Thread Scheduling:
- **Occupancy:** The number of active threads per SM relative to the maximum number that can be supported. Maximizing occupancy ensures the GPU's compute resources are fully utilized.
- **Thread Block Size:** The number of threads per block can impact scheduling. A common approach is to use 256 or 512 threads per block, but this can depend on the problem and GPU architecture.
- **Latency Hiding:** By having threads execute in parallel, CUDA can hide memory access latency, improving overall throughput.

## Factors Influencing Thread Scheduling:
- **Thread Divergence:** Occurs when threads in the same warp (32 threads) follow different execution paths. This can reduce performance since the GPU must serialize the divergent paths.
- **Memory Access Patterns:** Coalesced memory accesses (where threads access contiguous memory locations) help minimize latency and improve throughput.
- **Register Usage:** Too many registers per thread can reduce the number of threads per block and impact overall performance.

---

# Matrix Multiplication (MatMul) in CUDA

Matrix multiplication is a fundamental operation in linear algebra, widely used in scientific computing, machine learning, and graphics processing. Implementing matrix multiplication on the GPU using CUDA allows for significant performance improvements.

## Concept:
Given two matrices:
- **Matrix A** of size `MxN`
- **Matrix B** of size `NxP`

The result of the matrix multiplication, **Matrix C**, will be of size `MxP`. Each element `C[i,j]` of matrix C is calculated as the dot product of the i-th row of matrix A and the j-th column of matrix B:

