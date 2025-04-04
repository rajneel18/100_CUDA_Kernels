#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024             
#define TIMESTEPS 1000     
#define DT 0.1f            
#define TAU 10.0f          
#define V_REST -65.0f      
#define V_RESET -70.0f     
#define V_THRESH -50.0f    
#define R 1.0f            

// Kernel to simulate LIF neurons
__global__ void lif_neuron_kernel(float* v, int* spikes, float* input_current, int n, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float I = input_current[i]; // constant input per neuron

        // Update membrane potential
        v[i] += ((- (v[i] - V_REST) + R * I) * (DT / TAU));

        // Check for spike
        if (v[i] >= V_THRESH) {
            v[i] = V_RESET;
            spikes[i * TIMESTEPS + t] = 1;
        } else {
            spikes[i * TIMESTEPS + t] = 0;
        }
    }
}

int main() {
    float *d_v, *d_input_current;
    int *d_spikes;

    float *h_v = (float*)malloc(N * sizeof(float));
    float *h_input_current = (float*)malloc(N * sizeof(float));
    int *h_spikes = (int*)malloc(N * TIMESTEPS * sizeof(int));

    // Initialize voltages and inputs
    for (int i = 0; i < N; i++) {
        h_v[i] = V_REST;
        h_input_current[i] = 15.0f + (rand() % 10); // random input current
    }

    cudaMalloc(&d_v, N * sizeof(float));
    cudaMalloc(&d_input_current, N * sizeof(float));
    cudaMalloc(&d_spikes, N * TIMESTEPS * sizeof(int));

    cudaMemcpy(d_v, h_v, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_current, h_input_current, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    for (int t = 0; t < TIMESTEPS; t++) {
        lif_neuron_kernel<<<gridSize, blockSize>>>(d_v, d_spikes, d_input_current, N, t);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_spikes, d_spikes, N * TIMESTEPS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print spike raster for first 10 neurons
    for (int i = 0; i < 10; i++) {
        printf("Neuron %d: ", i);
        for (int t = 0; t < TIMESTEPS; t++) {
            printf("%d", h_spikes[i * TIMESTEPS + t]);
        }
        printf("\n");
    }

    cudaFree(d_v);
    cudaFree(d_input_current);
    cudaFree(d_spikes);
    free(h_v);
    free(h_input_current);
    free(h_spikes);

    return 0;
}

