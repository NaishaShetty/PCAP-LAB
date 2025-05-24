#include <stdio.h>
#include <cuda.h>

#define N 3  // Number of rows
#define M 3  // Number of columns

// Kernel for element-wise addition
__global__ void add_matrices_elementwise(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * M + col;
    if (row < N && col < M)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int a[N][M] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    int b[N][M] = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };
    int c[N][M];

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N*M*sizeof(int));
    cudaMalloc((void**)&d_b, N*M*sizeof(int));
    cudaMalloc((void**)&d_c, N*M*sizeof(int));

    cudaMemcpy(d_a, a, N*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*M*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((M + 15)/16, (N + 15)/16);

    add_matrices_elementwise<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N*M*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix (C = A + B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++)
            printf("%d ", c[i][j]);
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

