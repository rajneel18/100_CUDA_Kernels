#include <cuda_runtime.h>
#include <stdio.h>

#define N 16  
#define THREADS_PER_BLOCK 8

__device__ void merge(int *arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = new int[n1];
    int *R = new int[n2];

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

__global__ void mergeSortIterative(int *arr, int n) {
    int width = 2;  // Size of the sub-arrays to merge
    while (width <= n) {
        int start = blockIdx.x * blockDim.x + threadIdx.x;
        int left = start * width;
        if (left < n) {
            int mid = min(left + (width / 2) - 1, n - 1);
            int right = min(left + width - 1, n - 1);
            if (mid < right) {
                merge(arr, left, mid, right);
            }
        }
        width *= 2;
        __syncthreads();
    }
}

int main() {
    int h_arr[N] = {12, 11, 13, 5, 6, 7, 15, 9, 1, 0, 8, 14, 4, 10, 2, 3};  // Example array
    int *d_arr;

    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mergeSortIterative<<<numBlocks, THREADS_PER_BLOCK>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
