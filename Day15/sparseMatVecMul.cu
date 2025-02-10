#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel for SpMV using CSR format
__global__ void spmvCSR(int *rowPtr, int *colIdx, float *values, float *x, float *y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
            sum += values[j] * x[colIdx[j]];
        }
        y[row] = sum;
    }
}

int main() {
    int numRows = 4;
    int numCols = 4;
    int nnz = 8;  // Number of non-zero elements

    // CSR format data
    int h_rowPtr[] = {0, 2, 4, 6, 8};
    int h_colIdx[] = {0, 1, 1, 2, 2, 3, 3, 0};
    float h_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_x[] = {1.0, 2.0, 3.0, 4.0};
    float h_y[numRows] = {0.0};

    
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;
    cudaMalloc((void **)&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc((void **)&d_colIdx, nnz * sizeof(int));
    cudaMalloc((void **)&d_values, nnz * sizeof(float));
    cudaMalloc((void **)&d_x, numCols * sizeof(float));
    cudaMalloc((void **)&d_y, numRows * sizeof(float));

    
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numCols * sizeof(float), cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numRows + threadsPerBlock - 1) / threadsPerBlock;

    
    spmvCSR<<<blocksPerGrid, threadsPerBlock>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, numRows);

    
    cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    
    printf("Result Vector y:\n");
    for (int i = 0; i < numRows; ++i) {
        printf("%f ", h_y[i]);
    }
    printf("\n");

    
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
