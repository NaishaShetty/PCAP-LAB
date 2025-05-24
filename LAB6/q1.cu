#include <iostream>
#include <cuda_runtime.h>

#define MASK_WIDTH 5
#define TILE_WIDTH 16

// Input mask (device constant memory for fast access)
__constant__ float d_M[MASK_WIDTH];

// CUDA Kernel for 1D convolution
__global__ void convolution_1D(float* d_N, float* d_P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float P_val = 0.0f;
    int r = MASK_WIDTH / 2;

    for (int j = -r; j <= r; ++j) {
        int n_idx = i + j;
        float n_val = (n_idx >= 0 && n_idx < width) ? d_N[n_idx] : 0.0f;
        float m_val = d_M[j + r];
        P_val += n_val * m_val;
    }

    if (i < width)
        d_P[i] = P_val;
}

int main() {
    const int width = 20;
    const int mask_width = MASK_WIDTH;
    const int size = width * sizeof(float);

    float h_N[width], h_P[width], h_M[mask_width] = {3, 4, 5, 4, 3};

    // Initialize input
    for (int i = 0; i < width; ++i)
        h_N[i] = i + 1;

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, h_M, mask_width * sizeof(float));

    // Kernel launch
    int block_size = TILE_WIDTH;
    int grid_size = (width + block_size - 1) / block_size;
    convolution_1D<<<grid_size, block_size>>>(d_N, d_P, width);

    // Copy result back
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Input N: ";
    for (int i = 0; i < width; ++i) std::cout << h_N[i] << " ";
    std::cout << "\nOutput P: ";
    for (int i = 0; i < width; ++i) std::cout << h_P[i] << " ";
    std::cout << "\n";

    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}

