#include <stdio.h>
#include <cuda_runtime.h>

#define N 512 

__global__ void vectorAdd(int *a, int *b, int *c) {
    int i = threadIdx.x; // Each thread handles one element
    c[i] = a[i] + b[i];
}

int main() {
    int a[N], b[N], c[N];         // host arrays
    int *d_a, *d_b, *d_c;         // device arrays
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy input vectors to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and N threads
    vectorAdd<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print some of the results
    printf("Vector addition result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

