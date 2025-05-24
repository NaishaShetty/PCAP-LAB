#include <stdio.h>
#include <string.h>
#include <ctype.h> //checking if a character is uppercase, lowercase, a digit, etc.
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    char str[] = "HELLO";  // Input string (all processes will have the same copy)
    int len = strlen(str);

    MPI_Init(&argc, &argv);                     // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Get this process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);       // Get total number of processes

    if (size != len) {
        if (rank == 0)
            printf("Error: Number of processes must match string length (%d)\n", len);
        MPI_Finalize();
        return 0;
    }

    char ch = str[rank];  // Get character at index = rank

    // Toggle the character
    if (isupper(ch)) {
        ch = tolower(ch);
    } else if (islower(ch)) {
        ch = toupper(ch);
    }

    // Print result from each process
    printf("Process %d toggled character '%c' -> '%c'\n", rank, str[rank], ch);

    MPI_Finalize();  // Finalize MPI
    return 0;
}

