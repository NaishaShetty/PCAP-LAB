#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Array size

// CUDA kernel to find the index of the minimum element in a range
__global__ void find_min_index(float* d_arr, int start, int n, int* d_min_index) {
    extern __shared__ float shared_data[]; 
    float* s_vals = shared_data;                 // Shared array for values
    int* s_indices = (int*)&s_vals[blockDim.x];  // Shared array for indices

    int tid = threadIdx.x;
    int idx = start + tid;

    // Load values and indices or set to max if out of range
    if (idx < n) {
        s_vals[tid] = d_arr[idx];
        s_indices[tid] = idx;
    } else {
        s_vals[tid] = FLT_MAX;
        s_indices[tid] = -1;
    }
    __syncthreads();

    // Reduction to find minimum value and corresponding index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_vals[tid + stride] < s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_indices[tid] = s_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result to global memory
    if (tid == 0) {
        *d_min_index = s_indices[0];
    }
}


int main() {
    float h_arr[N] = {7, 3, 9, 2, 6, 1, 5, 4, 8, 0, 11, 10, 15, 14, 13, 12};
    float* d_arr;
    int* d_min_index;
    int min_index;

    cudaMalloc(&d_arr, N * sizeof(float));
    cudaMalloc(&d_min_index, sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < N - 1; ++i) {
       int threads = N - i;
       size_t shared_mem_size = threads * (sizeof(float) + sizeof(int));
       find_min_index<<<1, threads, shared_mem_size>>>(d_arr, i, N, d_min_index);


        // Swap h_arr[i] and h_arr[min_index] on the device
        float temp;
        cudaMemcpy(&temp, &d_arr[i], sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&d_arr[i], &d_arr[min_index], sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&d_arr[min_index], &temp, sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sorted array: ";
    for (int i = 0; i < N; ++i)
        std::cout << h_arr[i] << " ";
    std::cout << "\n";

    cudaFree(d_arr);
    cudaFree(d_min_index);
    return 0;
}

