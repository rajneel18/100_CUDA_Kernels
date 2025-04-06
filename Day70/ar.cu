#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define SEQ_COUNT 1   // Just 1 sequence for testing
#define P 5           // Order of the AR model
#define T 10          // Length of the input time series
#define F 20          // Forecast steps

__global__ void ar_forecast_kernel(float* past_data, float* coeffs, float* forecast, int p, int f, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < SEQ_COUNT) {
        float history[P];
        int offset = idx * t;

        // Copy the last P values into history
        for (int i = 0; i < p; ++i) {
            history[i] = past_data[offset + t - p + i];
        }

        for (int step = 0; step < f; ++step) {
            float next_val = 0.0f;
            for (int j = 0; j < p; ++j) {
                next_val += coeffs[j] * history[P - 1 - j];
            }

            // Store the prediction
            forecast[idx * f + step] = next_val;

            // Shift history and add new value
            for (int j = 0; j < P - 1; ++j) {
                history[j] = history[j + 1];
            }
            history[P - 1] = next_val;
        }
    }
}

int main() {
    float h_data[SEQ_COUNT * T] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}; // Test input
    float h_coeffs[P] = {0.1, 0.15, 0.2, 0.25, 0.3}; // AR coefficients
    float h_forecast[SEQ_COUNT * F] = {0};

    float *d_data, *d_coeffs, *d_forecast;

    cudaMalloc(&d_data, sizeof(h_data));
    cudaMalloc(&d_coeffs, sizeof(h_coeffs));
    cudaMalloc(&d_forecast, sizeof(h_forecast));

    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeffs, h_coeffs, sizeof(h_coeffs), cudaMemcpyHostToDevice);

    ar_forecast_kernel<<<1, 1>>>(d_data, d_coeffs, d_forecast, P, F, T);
    cudaDeviceSynchronize();

    cudaMemcpy(h_forecast, d_forecast, sizeof(h_forecast), cudaMemcpyDeviceToHost);

    printf("Last %d values of sequence 0 used for prediction:\n", P);
    for (int i = T - P; i < T; ++i) {
        printf("%.2f ", h_data[i]);
    }
    printf("\n\nForecasted values:\n");
    for (int i = 0; i < F; ++i) {
        printf("Step %d: %.4f\n", i + 1, h_forecast[i]);
    }

    cudaFree(d_data);
    cudaFree(d_coeffs);
    cudaFree(d_forecast);

    return 0;
}
