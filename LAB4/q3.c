#include <stdio.h>
#include <mpi.h>

#define SIZE 4

int main(int argc, char *argv[]) {
    int rank, size;
    int input[SIZE][SIZE];
    int row0[SIZE];
    int local_input[SIZE];
    int local_result[SIZE];
    int output[SIZE][SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < SIZE) {
        if (rank == 0)
            printf("Run program with %d processes.\n", SIZE);
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter 4x4 matrix row-wise:\n");
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                scanf("%d", &input[i][j]);
    }

    // Broadcast first row to all processes
    MPI_Bcast(input[0], SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter the rows to all processes
    MPI_Scatter(input, SIZE, MPI_INT, local_input, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local result
    for (int j = 0; j < SIZE; j++) {
        if (rank == 0) {
            local_result[j] = input[0][j];  // first row unchanged
        } else {
            local_result[j] = input[0][j] + local_input[j];
        }
    }

    // Gather local results back to root
    MPI_Gather(local_result, SIZE, MPI_INT, output, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nOutput matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                printf("%d ", output[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}

