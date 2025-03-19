#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>

#define SAMPLE_RATE 44100  // Audio sample rate (Hz)
#define N 1024             // FFT size (must be power of 2)
#define PI 3.14159265358979f

// Generate a sample sine wave signal
void generateSignal(float *signal, int size, float frequency, float sampleRate) {
    for (int i = 0; i < size; i++) {
        signal[i] = sinf(2.0f * PI * frequency * i / sampleRate);
    }
}

// Apply a simple noise filter (e.g., zeroing out high frequencies)
__global__ void filterFrequencyDomain(cufftComplex *data, int size, float cutoff) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float freq = idx * SAMPLE_RATE / size;  
        if (freq > cutoff) {
            data[idx].x = 0.0f;  // Zero out high-frequency components
            data[idx].y = 0.0f;
        }
    }
}

int main() {
    float *h_signal, *h_output;
    cufftComplex *d_signal;
    
    // Allocate host memory
    h_signal = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Generate test signal (sine wave of 1000 Hz)
    generateSignal(h_signal, N, 1000.0f, SAMPLE_RATE);

    // Allocate device memory
    cudaMalloc((void**)&d_signal, N * sizeof(cufftComplex));

    // Copy data to device
    cudaMemcpy(d_signal, h_signal, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create FFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_R2C, 1);

    // Perform forward FFT (Time domain → Frequency domain)
    cufftExecR2C(plan, (cufftReal*)d_signal, d_signal);

    // Apply a simple low-pass filter
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    filterFrequencyDomain<<<blocksPerGrid, threadsPerBlock>>>(d_signal, N, 2000.0f);  // Cutoff at 2kHz

    // Perform inverse FFT (Frequency domain → Time domain)
    cufftExecC2R(plan, d_signal, (cufftReal*)d_signal);

    cudaMemcpy(h_output, d_signal, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalize result (cuFFT scales incorrectly)
    for (int i = 0; i < N; i++) {
        h_output[i] /= N;
    }

    printf("Filtered Signal (First 10 Samples):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n", h_output[i]);
    }

    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);
    free(h_output);

    return 0;
}
