#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to compute factorial
long long factorial(int n) {
    long long result = 1;
    for (int i = 2; i <= n; i++)
        result *= i;
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int number;
    long long result;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (rank == 0) {
        int *values = (int *)malloc(size * sizeof(int));
        long long sum = 0;

        // Input N values
        printf("Enter %d integers:\n", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &values[i]);
        }

        // Send one value to each process (including self)
        for (int i = 0; i < size; i++) {
            if (i == 0) {
                number = values[0];
            } else {
                MPI_Send(&values[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }

        // Compute factorial in root
        result = factorial(number);
        sum += result;

        // Receive results from other processes
        for (int i = 1; i < size; i++) {
            MPI_Recv(&result, 1, MPI_LONG_LONG, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            sum += result;
        }

        printf("Sum of all factorials: %lld\n", sum);
        free(values);
    } else {
        // Receive value from root
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Compute factorial
        result = factorial(number);

        // Send back result to root
        MPI_Send(&result, 1, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}

