# Problems in CUDA
1. **Overhead**: 
   - CUDA programming can introduce overhead in setting up and managing GPU resources.
   - For small data sizes or simple tasks, the overhead might result in slower performance than sequential CPU processing.

2. **Memory Access Speed**:
   - The speed of CUDA programs can be limited by memory access speed rather than computation speed.
   - Applications restricted by memory access are called **memory-bound**.

3. **Characteristics of Data**:
   - To achieve high performance, data must be large enough to justify the GPU overhead.
   - The data should exhibit high **parallelism** for effective utilization of GPU cores.

# Data Parallelism in CUDA
- CUDA excels at **data parallelism**, where the same operation is applied to multiple data elements simultaneously.
- Examples:
  - Image processing (pixel-wise operations).
  - Vector and matrix computations.

# What Does CUDA API Do?
1. **Memory Management**:
   - Allocate memory on the GPU (device).
   - Transfer data from host (CPU) to device (GPU).
   - Transfer results from device to host after processing.
   - Free allocated memory on the device.

2. **Kernel Launch**:
   - Define and execute kernel functions (functions executed on the GPU).

3. **Thread Management**:
   - Organize threads into grids and blocks for parallel execution.

# Flow of Vector Addition Kernel
1. **Host Code**:
   - Allocate memory on the host and device.
   - Transfer input vectors from host to device memory.
   - Launch the kernel to perform vector addition on the GPU.

2. **Kernel Code**:
   - Each thread calculates one element of the output vector.
   - Example: `C[i] = A[i] + B[i]` for index `i`.

3. **Host Code** (continued):
   - Transfer the result vector from device to host memory.
   - Free the allocated device memory.
