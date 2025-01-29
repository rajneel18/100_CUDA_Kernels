#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(){
    int *d_A;
    size_t size = 100000000000 * sizeof(int);

    cudaError_t err = cudaMalloc((void**)&d_A, size);

    if (err!= cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    else{
        printf("Memory allocated successfully.\n");
    }

    cudaFree(d_A);

    return 0;

}