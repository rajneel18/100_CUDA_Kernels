#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define INPUT_DIM 3     // Dimensionality of input vectors
#define GRID_SIZE 10    // SOM grid size (10x10 neurons)
#define NUM_NEURONS (GRID_SIZE * GRID_SIZE)
#define LEARNING_RATE 0.1
#define NUM_ITERS 1000

// Kernel to initialize weights using random values
__global__ void init_weights(float *weights, curandState *states, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NUM_NEURONS * INPUT_DIM) {
        curand_init(seed, idx, 0, &states[idx]);
        weights[idx] = curand_uniform(&states[idx]);
    }
}

// Kernel to compute Euclidean distance for each neuron
__global__ void compute_distances(float *weights, float *input, float *distances) {
    int neuron_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (neuron_idx < NUM_NEURONS) {
        float dist = 0.0;
        for (int d = 0; d < INPUT_DIM; d++) {
            float diff = weights[neuron_idx * INPUT_DIM + d] - input[d];
            dist += diff * diff;
        }
        distances[neuron_idx] = dist;
    }
}

// Kernel to update the weights of the winner neuron and neighbors
__global__ void update_weights(float *weights, float *input, int winner_idx, float learning_rate) {
    int neuron_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (neuron_idx < NUM_NEURONS) {
        float influence = expf(-((neuron_idx - winner_idx) * (neuron_idx - winner_idx)) / (2.0f));
        for (int d = 0; d < INPUT_DIM; d++) {
            weights[neuron_idx * INPUT_DIM + d] += learning_rate * influence * 
                (input[d] - weights[neuron_idx * INPUT_DIM + d]);
        }
    }
}

// Host function to find the best matching unit (BMU)
int find_bmu(float *distances, int num_neurons) {
    int bmu_idx = 0;
    float min_dist = distances[0];
    for (int i = 1; i < num_neurons; i++) {
        if (distances[i] < min_dist) {
            min_dist = distances[i];
            bmu_idx = i;
        }
    }
    return bmu_idx;
}

int main() {
    float *d_weights, *d_input, *d_distances;
    curandState *d_states;

    cudaMalloc(&d_weights, NUM_NEURONS * INPUT_DIM * sizeof(float));
    cudaMalloc(&d_input, INPUT_DIM * sizeof(float));
    cudaMalloc(&d_distances, NUM_NEURONS * sizeof(float));
    cudaMalloc(&d_states, NUM_NEURONS * INPUT_DIM * sizeof(curandState));

    init_weights<<<(NUM_NEURONS * INPUT_DIM + 255) / 256, 256>>>(d_weights, d_states, time(NULL));
    cudaDeviceSynchronize();

    float h_input[INPUT_DIM] = {0.5, 0.8, 0.3}; // Example input vector
    cudaMemcpy(d_input, h_input, INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    for (int iter = 0; iter < NUM_ITERS; iter++) {

        compute_distances<<<(NUM_NEURONS + 255) / 256, 256>>>(d_weights, d_input, d_distances);
        cudaDeviceSynchronize();

        float h_distances[NUM_NEURONS];
        cudaMemcpy(h_distances, d_distances, NUM_NEURONS * sizeof(float), cudaMemcpyDeviceToHost);

        int bmu_idx = find_bmu(h_distances, NUM_NEURONS);

        update_weights<<<(NUM_NEURONS + 255) / 256, 256>>>(d_weights, d_input, bmu_idx, LEARNING_RATE);
        cudaDeviceSynchronize();

        printf("Iteration %d: BMU = %d\n", iter, bmu_idx);
    }

    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_distances);
    cudaFree(d_states);

    return 0;
}
