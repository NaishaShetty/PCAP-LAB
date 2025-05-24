#include <stdio.h>
#include <mpi.h>

#define ROWS 3
#define COLS 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[ROWS][COLS];
    int element;
    int local_row[COLS];
    int local_count = 0, total_count;

    MPI_Init(&argc, &argv);                  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);    // Get number of processes

    if (size != 3) {
        if (rank == 0)
            printf("This program requires exactly 3 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // Root process reads the matrix and the element
    if (rank == 0) {
        printf("Enter a 3x3 matrix:\n");
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to search for: ");
        scanf("%d", &element);
    }

    // Broadcast the search element to all processes
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of the matrix to all processes
    MPI_Scatter(matrix, COLS, MPI_INT, local_row, COLS, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts occurrences in its row
    for (int i = 0; i < COLS; i++) {
        if (local_row[i] == element)
            local_count++;
    }

    // Reduce all local counts to the root process
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("The element %d occurred %d time(s) in the matrix.\n", element, total_count);
    }

    MPI_Finalize();
    return 0;
}

