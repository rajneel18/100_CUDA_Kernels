#include <stdio.h>
#include <curand_kernel.h>

#define NUM_NEURONS 1024
#define TIME_STEPS 100
#define THRESHOLD 1.0f
#define DECAY 0.95f

__global__ void init_random_states(curandState *states, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_NEURONS) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void spiking_neuron_sim(float *membrane_potential, int *spike_counts, curandState *states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_NEURONS) {
        curandState localState = states[idx];
        for (int t = 0; t < TIME_STEPS; t++) {
            float input_current = curand_uniform(&localState) * 0.1f;
            membrane_potential[idx] += input_current;
            if (membrane_potential[idx] >= THRESHOLD) {
                spike_counts[idx]++;
                membrane_potential[idx] = 0.0f;  // Reset after spike
            } else {
                membrane_potential[idx] *= DECAY;
            }
        }
        states[idx] = localState;
    }
}

int main() {
    float *d_membrane_potential;
    int *d_spike_counts;
    curandState *d_states;
    
    cudaMalloc(&d_membrane_potential, NUM_NEURONS * sizeof(float));
    cudaMalloc(&d_spike_counts, NUM_NEURONS * sizeof(int));
    cudaMalloc(&d_states, NUM_NEURONS * sizeof(curandState));
    
    cudaMemset(d_membrane_potential, 0, NUM_NEURONS * sizeof(float));
    cudaMemset(d_spike_counts, 0, NUM_NEURONS * sizeof(int));
    
    dim3 blockSize(256);
    dim3 gridSize((NUM_NEURONS + blockSize.x - 1) / blockSize.x);
    
    init_random_states<<<gridSize, blockSize>>>(d_states, time(NULL));
    spiking_neuron_sim<<<gridSize, blockSize>>>(d_membrane_potential, d_spike_counts, d_states);
    
    int *h_spike_counts = (int*)malloc(NUM_NEURONS * sizeof(int));
    cudaMemcpy(h_spike_counts, d_spike_counts, NUM_NEURONS * sizeof(int), cudaMemcpyDeviceToHost);
    
    int total_spikes = 0;
    for (int i = 0; i < NUM_NEURONS; i++) {
        total_spikes += h_spike_counts[i];
    }
    printf("Total spikes in the network: %d\n", total_spikes);
    
    free(h_spike_counts);
    cudaFree(d_membrane_potential);
    cudaFree(d_spike_counts);
    cudaFree(d_states);
    
    return 0;
}
