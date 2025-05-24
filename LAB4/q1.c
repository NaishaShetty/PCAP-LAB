#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute factorial
long long factorial(int n) {
    long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

// Custom error handler
void mpi_error_handler(MPI_Comm *comm, int *err_code, ...) {
    char error_string[BUFSIZ];
    int length_of_error_string;
    MPI_Error_string(*err_code, error_string, &length_of_error_string);
    fprintf(stderr, "MPI Error: %s\n", error_string);
    MPI_Abort(MPI_COMM_WORLD, *err_code);
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    long long local_fact, local_sum;

    // Initialize MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "Error initializing MPI.\n");
        exit(EXIT_FAILURE);
    }

    // Set custom error handler
    MPI_Errhandler err_handler;
    MPI_Comm_create_errhandler(mpi_error_handler, &err_handler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, err_handler);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get N from command line or default to number of processes
    if (rank == 0) {
        if (argc != 2) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        N = atoi(argv[1]);
        if (N < size) {
            fprintf(stderr, "Warning: N (%d) is less than number of processes (%d). Not all processes will be used.\n", N, size);
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank < N) {
        int i = rank + 1;
        local_fact = factorial(i);
    } else {
        local_fact = 0;
    }

    // Use MPI_Scan to compute prefix sum of factorials
    MPI_Scan(&local_fact, &local_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    // The last active process will print the final result
    if (rank == N - 1) {
        printf("Sum of factorials from 1! to %d! is: %lld\n", N, local_sum);
    }

    MPI_Errhandler_free(&err_handler);
    MPI_Finalize();
    return 0;
}

