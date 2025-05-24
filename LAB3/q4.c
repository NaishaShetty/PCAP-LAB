#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char *S1 = NULL, *S2 = NULL;
    int str_len, chunk_size;
    char *chunk1, *chunk2;
    char *interleaved_chunk, *final_result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        char temp1[1000], temp2[1000];
        printf("Enter String S1: ");
        scanf("%s", temp1);
        printf("Enter String S2: ");
        scanf("%s", temp2);

        str_len = strlen(temp1);

        if (str_len != strlen(temp2) || str_len % size != 0) {
            printf("Error: Strings must be of equal length and divisible by number of processes (%d).\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = str_len / size;

        S1 = (char *)malloc(str_len * sizeof(char));
        S2 = (char *)malloc(str_len * sizeof(char));
        strcpy(S1, temp1);
        strcpy(S2, temp2);
    }

    // Broadcast string length and chunk size
    MPI_Bcast(&str_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = str_len / size;

    // Allocate space for each process's chunk
    chunk1 = (char *)malloc(chunk_size * sizeof(char));
    chunk2 = (char *)malloc(chunk_size * sizeof(char));
    interleaved_chunk = (char *)malloc(2 * chunk_size * sizeof(char));

    // Scatter the strings to all processes
    MPI_Scatter(S1, chunk_size, MPI_CHAR, chunk1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, chunk2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Interleave characters
    for (int i = 0; i < chunk_size; i++) {
        interleaved_chunk[2 * i] = chunk1[i];
        interleaved_chunk[2 * i + 1] = chunk2[i];
    }

    // Gather interleaved parts to root
    if (rank == 0) {
        final_result = (char *)malloc(2 * str_len * sizeof(char));
    }

    MPI_Gather(interleaved_chunk, 2 * chunk_size, MPI_CHAR, final_result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root prints final result
    if (rank == 0) {
        final_result[2 * str_len] = '\0'; // null-terminate
        printf("Resultant String: %s\n", final_result);
        free(S1);
        free(S2);
        free(final_result);
    }

    free(chunk1);
    free(chunk2);
    free(interleaved_chunk);

    MPI_Finalize();
    return 0;
}


