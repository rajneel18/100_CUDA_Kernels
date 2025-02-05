#include <stdio.h>
#include <cuda_runtime.h>

// ----------------- CONSTANT MEMORY -------------------
__constant__ int constData[256];

// ----------------- TEXTURE MEMORY (Modern API) -------------------
cudaTextureObject_t texObj; // NEW API

// ----------------- KERNEL USING GLOBAL MEMORY -------------------
__global__ void globalMemoryKernel(int *d_data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    d_data[tid] += 10;  // Each thread modifies its own index
}

// ----------------- KERNEL USING SHARED MEMORY -------------------
__global__ void sharedMemoryKernel(int *d_data) {
    __shared__ int sharedData[256]; // Declare shared memory

    int tid = threadIdx.x;
    sharedData[tid] = d_data[tid]; // Load from global to shared
    __syncthreads(); // Synchronize all threads

    d_data[tid] = sharedData[tid] * 2; // Perform operation and store result
}

// ----------------- KERNEL USING CONSTANT MEMORY -------------------
__global__ void constantMemoryKernel(int *d_data) {
    int tid = threadIdx.x;
    d_data[tid] += constData[tid]; // Access constant memory
}

// ----------------- KERNEL USING TEXTURE MEMORY (Modern API) -------------------
__global__ void textureMemoryKernel(cudaTextureObject_t texObj, int *d_output) {
    int tid = threadIdx.x;
    d_output[tid] = tex1Dfetch<int>(texObj, tid); // Fetch using texture object
}

// ----------------- KERNEL USING REGISTER MEMORY -------------------
__global__ void registerMemoryKernel(int *d_data) {
    int tid = threadIdx.x;
    int regValue = d_data[tid]; // Register variable (fast)
    regValue *= 2;
    d_data[tid] = regValue;
}

// ----------------- HOST CODE (MAIN FUNCTION) -------------------
int main() {
    const int SIZE = 256;
    int h_data[SIZE], h_constData[SIZE];
    int *d_data;

    // Initialize host data
    for (int i = 0; i < SIZE; i++) {
        h_data[i] = i;
        h_constData[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_data, SIZE * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constData, h_constData, SIZE * sizeof(int)); // Copy to constant memory

    // ----------------- SETUP TEXTURE MEMORY -------------------
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.sizeInBytes = SIZE * sizeof(int);
    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc.res.linear.desc.x = 32; // 32-bit integers

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Launch kernels
    globalMemoryKernel<<<1, SIZE>>>(d_data);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Global Memory (First 5): ");
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]); printf("\n");

    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    sharedMemoryKernel<<<1, SIZE>>>(d_data);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Shared Memory (First 5): ");
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]); printf("\n");

    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    constantMemoryKernel<<<1, SIZE>>>(d_data);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Constant Memory (First 5): ");
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]); printf("\n");

    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    textureMemoryKernel<<<1, SIZE>>>(texObj, d_data);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Texture Memory (First 5): ");
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]); printf("\n");

    cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    registerMemoryKernel<<<1, SIZE>>>(d_data);
    cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Register Memory (First 5): ");
    for (int i = 0; i < 5; i++) printf("%d ", h_data[i]); printf("\n");

    // Cleanup
    cudaDestroyTextureObject(texObj); // Destroy texture object
    cudaFree(d_data);

    return 0;
}
