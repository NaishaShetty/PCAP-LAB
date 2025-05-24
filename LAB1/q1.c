#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    const int x = 2;  

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Compute power
    double result = pow(x, rank);

    // Print result
    printf("Process %d: %d^%d = %.0f\n", rank, x, rank, result);

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

