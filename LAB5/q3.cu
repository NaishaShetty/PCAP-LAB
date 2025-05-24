#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 512  
#define THREADS_PER_BLOCK 256

// CUDA kernel to compute sine of each element
__global__ void computeSine(float *angles, float *results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        results[i] = __sinf(angles[i]);  
    }
}

int main() {
    float angles[N], results[N];    // Host arrays
    float *d_angles, *d_results;    // Device arrays
    size_t size = N * sizeof(float);

    // Initialize input array with some angles in radians
    for (int i = 0; i < N; i++) {
        angles[i] = i * 0.01f;  // Angles from 0 to ~5.12 radians
    }

    // Allocate device memory
    cudaMalloc((void **)&d_angles, size);
    cudaMalloc((void **)&d_results, size);

    // Copy input to device
    cudaMemcpy(d_angles, angles, size, cudaMemcpyHostToDevice);

    // Launch kernel with dynamic block calculation
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    computeSine<<<blocks, THREADS_PER_BLOCK>>>(d_angles, d_results, N);

    // Copy result back to host
    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);

    // Print first 10 results
    printf("Sine of angles (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("sin(%.4f) = %.4f\n", angles[i], results[i]);
    }

    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_results);

    return 0;
}

