#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to check if a character is a vowel
int is_vowel(char c) {
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char *argv[]) {
    int rank, size;
    char *input_string = NULL;
    int *counts = NULL;
    char *chunk;
    int chunk_size;
    int local_count = 0;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (rank == 0) {
        // Read string from user
        char temp[1000];
        printf("Enter a string (length divisible by %d): ", size);
        scanf("%s", temp);

        int len = strlen(temp);

        if (len % size != 0) {
            printf("Error: String length must be divisible by number of processes (%d).\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = len / size;

        input_string = (char *)malloc(len * sizeof(char));
        strcpy(input_string, temp);
        counts = (int *)malloc(size * sizeof(int));
    }

    // Broadcast chunk size to all processes
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local chunk
    chunk = (char *)malloc((chunk_size + 1) * sizeof(char));

    // Scatter the string chunks to all processes
    MPI_Scatter(input_string, chunk_size, MPI_CHAR, chunk, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    chunk[chunk_size] = '\0'; // Null-terminate for safety

    // Each process counts non-vowels
    local_count = 0;
    for (int i = 0; i < chunk_size; i++) {
        if (!is_vowel(chunk[i])) {
            local_count++;
        }
    }

    // Gather counts to root process
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root prints individual and total counts
    if (rank == 0) {
        int total = 0;
        printf("\nNon-vowel counts by each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d\n", i, counts[i]);
            total += counts[i];
        }
        printf("Total number of non-vowels: %d\n", total);
        free(input_string);
        free(counts);
    }

    free(chunk);
    MPI_Finalize(); // Finalize MPI
    return 0;
}

