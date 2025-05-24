#include <stdio.h>
#include <mpi.h>

// Function to compute factorial
long long factorial(int n) {
    long long fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

// Function to compute nth Fibonacci number
long long fibonacci(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    long long a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                     // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get total number of processes

    if (rank % 2 == 0) {
        // Even-ranked process: compute factorial
        long long fact = factorial(rank);
        printf("Process %d (even): Factorial(%d) = %lld\n", rank, rank, fact);
    } else {
        // Odd-ranked process: compute Fibonacci
        long long fib = fibonacci(rank);
        printf("Process %d (odd): Fibonacci(%d) = %lld\n", rank, rank, fib);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}

