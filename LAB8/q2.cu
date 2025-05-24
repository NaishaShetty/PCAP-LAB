#include <stdio.h>
#include <cuda.h>

#define N 3  // Rows in A and C
#define M 3  // Columns in A and Rows in B
#define P 3  // Columns in B and C

// Kernel for element-wise matrix multiplication
__global__ void multiply_matrices_elementwise(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        int sum = 0;
        for (int k = 0; k < M; k++) {
            sum += a[row * M + k] * b[k * P + col];
        }
        c[row * P + col] = sum;
    }
}

int main() {
    int a[N][M] = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    int b[M][P] = { {9, 8, 7}, {6, 5, 4}, {3, 2, 1} };
    int c[N][P];

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, N*M*sizeof(int));
    cudaMalloc((void**)&d_b, M*P*sizeof(int));
    cudaMalloc((void**)&d_c, N*P*sizeof(int));

    cudaMemcpy(d_a, a, N*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, M*P*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + 15)/16, (N + 15)/16);

    multiply_matrices_elementwise<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, N*P*sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix (C = A * B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++)
            printf("%d ", c[i][j]);
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

