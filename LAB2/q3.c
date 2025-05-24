#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *data = NULL;
    int value;
    int bufsize;
    void *buffer;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    // Calculate buffer size and attach
    bufsize = size * (MPI_BSEND_OVERHEAD + sizeof(int));
    buffer = malloc(bufsize);
    MPI_Buffer_attach(buffer, bufsize);

    if (rank == 0) {
        // Root process
        data = (int *)malloc(size * sizeof(int));

        printf("Enter %d integers:\n", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &data[i]);
        }

        // Send one value to each process (including itself)
        for (int i = 0; i < size; i++) {
            MPI_Bsend(&data[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        free(data);
    }

    // All processes receive one value from root
    MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank % 2 == 0) {
        // Even-ranked process: square the number
        printf("Process %d received %d, square = %d\n", rank, value, value * value);
    } else {
        // Odd-ranked process: cube the number
        printf("Process %d received %d, cube = %d\n", rank, value, value * value * value);
    }

    MPI_Buffer_detach(&buffer, &bufsize);
    free(buffer);

    MPI_Finalize(); // Finalize MPI
    return 0;
}

