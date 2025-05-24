#include <cuda_runtime.h>
#include <stdio.h>

__global__ void oddEvenTranspositionSortStep(int *arr, int n, int phase) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Calculate the index of the pair to compare
    int i = 2 * idx + phase;

    if (i + 1 < n) {
        if (arr[i] > arr[i + 1]) {
            // Swap elements if they are out of order
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

void oddEvenTranspositionSort(int *h_arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);

    // Allocate device memory and copy the input array
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Perform n phases (odd-even iterations)
    for (int phase = 0; phase < n; phase++) {
        // phase % 2 == 0 -> even phase, else odd phase
        int currentPhase = phase % 2;
        int pairs = (n - currentPhase) / 2;

        // Adjust grid size for the number of pairs
        blocksPerGrid = (pairs + threadsPerBlock - 1) / threadsPerBlock;

        oddEvenTranspositionSortStep<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, currentPhase);
        cudaDeviceSynchronize();
    }

    // Copy sorted array back to host
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int h_arr[] = {5, 3, 8, 4, 2, 7, 1, 6};
    int n = sizeof(h_arr) / sizeof(h_arr[0]);

    printf("Original array:\n");
    for (int i = 0; i < n; i++) printf("%d ", h_arr[i]);
    printf("\n");

    oddEvenTranspositionSort(h_arr, n);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) printf("%d ", h_arr[i]);
    printf("\n");

    return 0;
}

