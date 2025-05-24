#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, value;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (rank == 0) {
        // Root process reads an integer
        printf("Enter an integer value: ");
        scanf("%d", &value);

        value += 1; // Increment before sending
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // Send to Process 1

        // Receive from last process
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Root process received final value: %d\n", value);
    } else {
        // All other processes
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += 1;

        if (rank == size - 1) {
            // Last process sends back to root
            MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            // Intermediate process sends to next
            MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}

