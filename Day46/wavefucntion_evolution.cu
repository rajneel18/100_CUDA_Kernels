#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define N 1024       // Number of spatial points
#define DX 0.01      // Spatial step
#define DT 5e-7      // Time step
#define HBAR 1.0     // Planck's constant (normalized)
#define MASS 1.0     // Particle mass (normalized)
#define BLOCK_SIZE 256

typedef struct {
    double x;
    double y;
} complexd;

__device__ complexd complex_add(complexd a, complexd b) {
    complexd c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

__device__ complexd complex_sub(complexd a, complexd b) {
    complexd c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

__device__ complexd complex_mul(double scalar, complexd a) {
    complexd c;
    c.x = scalar * a.x;
    c.y = scalar * a.y;
    return c;
}

__device__ complexd complex_mul_complex(complexd a, complexd b) {
    complexd c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__global__ void evolve_wavefunction(complexd *psi, complexd *psi_next, double *potential) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0 && i < N - 1) {
        complexd laplacian = complex_sub(
            complex_add(psi[i - 1], psi[i + 1]),
            complex_mul(2.0, psi[i])
        );
        laplacian = complex_mul(1.0 / (DX * DX), laplacian);

        complexd i_hbar;
        i_hbar.x = 0.0;
        i_hbar.y = HBAR;
        
        complexd term1 = complex_mul(DT / (2.0 * MASS), complex_mul_complex(i_hbar, laplacian));
        complexd term2 = complex_mul(DT / HBAR, complex_mul_complex(i_hbar, complex_mul(potential[i], psi[i])));
        
        psi_next[i] = complex_add(complex_sub(psi[i], term1), term2);
    }
}

int main() {
    complexd *d_psi, *d_psi_next;
    double *d_potential;
    complexd h_psi[N];
    double h_potential[N];
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int i = 0; i < N; i++) {
        double x = (i - N / 2) * DX;
        double envelope = exp(-x * x);
        h_psi[i].x = envelope * cos(5.0 * x);
        h_psi[i].y = envelope * sin(5.0 * x);
        h_potential[i] = 0.5 * x * x;
    }
    
    cudaMalloc(&d_psi, N * sizeof(complexd));
    cudaMalloc(&d_psi_next, N * sizeof(complexd));
    cudaMalloc(&d_potential, N * sizeof(double));
    
    cudaMemcpy(d_psi, h_psi, N * sizeof(complexd), cudaMemcpyHostToDevice);
    cudaMemcpy(d_potential, h_potential, N * sizeof(double), cudaMemcpyHostToDevice);
    
    for (int t = 0; t < 1000; t++) {
        evolve_wavefunction<<<blocks, BLOCK_SIZE>>>(d_psi, d_psi_next, d_potential);
        cudaDeviceSynchronize();
        complexd *temp = d_psi;
        d_psi = d_psi_next;
        d_psi_next = temp;
    }
    
    cudaMemcpy(h_psi, d_psi, N * sizeof(complexd), cudaMemcpyDeviceToHost);
    
    cudaFree(d_psi);
    cudaFree(d_psi_next);
    cudaFree(d_potential);
    
    for (int i = 0; i < N; i += N / 10) {
        printf("x: %f | Psi: (%f, %f)\n", (i - N / 2) * DX, h_psi[i].x, h_psi[i].y);
    }
    return 0;
}
