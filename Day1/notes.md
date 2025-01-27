
# CUDA Use Cases
1. **Graphics Rendering**: CUDA accelerates graphics rate, improving real-time rendering in video games and simulations.
2. **Crypto Mining**: CUDA powers parallel computations in cryptocurrency mining, speeding up hashing operations.
3. **3D Modeling**: CUDA enhances rendering speeds and real-time 3D model processing for animations and virtual environments.
4. **Deep Learning**: CUDA is widely used to train neural networks, leveraging GPU's parallel architecture for faster model training.

# CPU vs GPU Functions
- **CPU**: Executes functions sequentially (one task at a time).
- **GPU**: Executes **kernels** (a small program) in parallel, enabling simultaneous tasks over thousands of threads.

# CUDA Programming Structure
- **Host (h_a)**: The CPU memory where data is stored before being sent to the GPU.
- **Device (d_a)**: The GPU memory where data is processed and stored.

# CUDA Keywords and Functions
1. **`__global__`**: Specifies a kernel function that runs on the GPU.
2. **`__device__`**: Functions that run on the GPU but are called from other GPU functions.
3. **`__host__`**: Functions that run on the CPU.
4. **Memory Functions**: `cudaMalloc`, `cudaMemcpy`, `cudaFree` for allocating, copying, and freeing memory on the device.

# Grid and Thread Structure
1. **GridDim.x**: Specifies the grid size (number of blocks in the grid).
2. **BlockDim**: Defines the number of threads per block.
3. **ThreadIdx**: Identifies each thread in the block.

# SIMD (Single Instruction, Multiple Data)
- A parallel computing model where the same instruction is executed on multiple data points simultaneously.

# RGB to Greyscale Conversion (CUDA Example)
- **Host Code**:
  - Allocates memory for images (RGB and grayscale).
  - Copies data from host to device.
  - Calls the CUDA kernel for conversion.
  - Copies results back to the host.
  
- **Device Kernel**:
  - Computes the grayscale value for each pixel by applying the formula:
    \[
    	ext{Gray} = 0.2989 	imes 	ext{R} + 0.5870 	imes 	ext{G} + 0.1140 	imes 	ext{B}
    \]
  - Each thread processes one pixel.
