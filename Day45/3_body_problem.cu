#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define G 6.67430e-11f   // Gravitational constant
#define DT 1e-3f         // Time step
#define STEPS 1000       // Simulation steps
#define SOFTENING 1e-5f  // Softening factor
#define NUM_BODIES 1000  // Number of bodies

typedef struct {
    float3 position;
    float3 velocity;
    float mass;
} Body;

__device__ float3 computeAcceleration(Body* bodies, int id, int numBodies) {
    float3 acc = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < numBodies; i++) {
        if (i != id) {
            float3 r;
            r.x = bodies[i].position.x - bodies[id].position.x;
            r.y = bodies[i].position.y - bodies[id].position.y;
            r.z = bodies[i].position.z - bodies[id].position.z;
            float distSq = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING * SOFTENING;
            float dist = sqrtf(distSq);
            float force = G * bodies[i].mass / distSq;
            acc.x += force * r.x / dist;
            acc.y += force * r.y / dist;
            acc.z += force * r.z / dist;
        }
    }
    return acc;
}

__global__ void updateBodies(Body* d_bodies, int numBodies) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= numBodies) return;
    
    float3 acc = computeAcceleration(d_bodies, id, numBodies);
    
    d_bodies[id].velocity.x += acc.x * DT;
    d_bodies[id].velocity.y += acc.y * DT;
    d_bodies[id].velocity.z += acc.z * DT;
    
    d_bodies[id].position.x += d_bodies[id].velocity.x * DT;
    d_bodies[id].position.y += d_bodies[id].velocity.y * DT;
    d_bodies[id].position.z += d_bodies[id].velocity.z * DT;
}

void initializeBodies(Body* h_bodies, int numBodies) {
    for (int i = 0; i < numBodies; i++) {
        h_bodies[i].position.x = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        h_bodies[i].position.y = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        h_bodies[i].position.z = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;
        h_bodies[i].velocity.x = 0.0f;
        h_bodies[i].velocity.y = 0.0f;
        h_bodies[i].velocity.z = 0.0f;
        h_bodies[i].mass = 1.0e10f;
    }
}

int main() {
    Body* h_bodies = (Body*)malloc(NUM_BODIES * sizeof(Body));
    Body* d_bodies;
    
    initializeBodies(h_bodies, NUM_BODIES);
    cudaMalloc((void**)&d_bodies, NUM_BODIES * sizeof(Body));
    cudaMemcpy(d_bodies, h_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int numBlocks = (NUM_BODIES + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int i = 0; i < STEPS; i++) {
        updateBodies<<<numBlocks, threadsPerBlock>>>(d_bodies, NUM_BODIES);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(h_bodies, d_bodies, NUM_BODIES * sizeof(Body), cudaMemcpyDeviceToHost);
    cudaFree(d_bodies);
    
    printf("Final Positions:\n");
    for (int i = 0; i < 10; i++) {
        printf("Body %d: (%.2f, %.2f, %.2f)\n", i, h_bodies[i].position.x, h_bodies[i].position.y, h_bodies[i].position.z);
    }
    
    free(h_bodies);
    return 0;
}
