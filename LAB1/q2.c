#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Print message based on rank
    if (rank % 2 == 0) {
        printf("Process %d: Hello\n", rank);
    } else {
        printf("Process %d: World\n", rank);
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}


