#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

#define POP_SIZE 1024
#define CHROMO_LEN 16
#define MUTATION_RATE 0.1
#define GENERATIONS 10
#define BLOCK_SIZE 256

__device__ float fitness_function(int chromosome) {
    return -(chromosome * chromosome) + 10 * chromosome; // Example function
}

__global__ void initialize_population(int *population, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POP_SIZE) {
        curandState localState = states[idx];
        population[idx] = curand(&localState) % (1 << CHROMO_LEN); // Random binary chromosome
        states[idx] = localState;
    }
}

__global__ void evaluate_fitness(int *population, float *fitness) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POP_SIZE) {
        fitness[idx] = fitness_function(population[idx]);
    }
}

__global__ void mutate(int *population, curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < POP_SIZE) {
        curandState localState = states[idx];
        if (curand_uniform(&localState) < MUTATION_RATE) {
            int bit = curand(&localState) % CHROMO_LEN;
            population[idx] ^= (1 << bit); // Flip a random bit
        }
        states[idx] = localState;
    }
}

__global__ void setup_curand(curandState *states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, idx, 0, &states[idx]);
}

void genetic_algorithm() {
    int *d_population;
    float *d_fitness;
    curandState *d_states;
    int *h_population = (int *)malloc(POP_SIZE * sizeof(int));
    float *h_fitness = (float *)malloc(POP_SIZE * sizeof(float));

    cudaMalloc(&d_population, POP_SIZE * sizeof(int));
    cudaMalloc(&d_fitness, POP_SIZE * sizeof(float));
    cudaMalloc(&d_states, POP_SIZE * sizeof(curandState));

    setup_curand<<<(POP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states);
    initialize_population<<<(POP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_states);
    
    for (int gen = 0; gen < GENERATIONS; gen++) {
        evaluate_fitness<<<(POP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_fitness);
        cudaMemcpy(h_population, d_population, POP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fitness, d_fitness, POP_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        // Find the best fitness
        float best_fitness = -1e9;
        int best_chromo = 0;
        for (int i = 0; i < POP_SIZE; i++) {
            if (h_fitness[i] > best_fitness) {
                best_fitness = h_fitness[i];
                best_chromo = h_population[i];
            }
        }
        printf("Generation %d: Best Fitness = %.2f, Best Chromosome = %d\n", gen, best_fitness, best_chromo);

        mutate<<<(POP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_population, d_states);
    }

    cudaMemcpy(h_population, d_population, POP_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitness, d_fitness, POP_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nFinal Best Chromosome: %d, Fitness: %.2f\n", h_population[0], h_fitness[0]);

    cudaFree(d_population);
    cudaFree(d_fitness);
    cudaFree(d_states);
    free(h_population);
    free(h_fitness);
}

int main() {
    genetic_algorithm();
    return 0;
}
