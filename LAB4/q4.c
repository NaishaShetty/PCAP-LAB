#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char word[100];         // to store input word (max 100 chars)
    char ch, *local_str, *recv_buf;
    int N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Step 1: Root reads the input word
    if (rank == 0) {
        printf("Enter a word of length %d: ", size);
        scanf("%s", word);
        N = strlen(word);
        if (N != size) {
            printf("Error: Word length must be equal to number of processes (%d)\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Step 2: Broadcast the input word to all processes
    MPI_Bcast(word, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Step 3: Each process computes its repeated character (i+1 times)
    ch = word[rank];
    int repeat = rank + 1;
    local_str = (char *)malloc((repeat + 1) * sizeof(char));
    for (int i = 0; i < repeat; i++) {
        local_str[i] = ch;
    }
    local_str[repeat] = '\0';

    // Step 4: Gather lengths first
    int local_len = repeat;
    int *recv_counts = NULL, *displs = NULL;
    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
    }

    MPI_Gather(&local_len, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Step 5: Root computes displacements and total length
    int total_len = 0;
    if (rank == 0) {
        displs[0] = 0;
        total_len += recv_counts[0];
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + recv_counts[i - 1];
            total_len += recv_counts[i];
        }
        recv_buf = (char *)malloc((total_len + 1) * sizeof(char));
    }

    // Step 6: Gather all parts of the output word
    MPI_Gatherv(local_str, local_len, MPI_CHAR,
                recv_buf, recv_counts, displs, MPI_CHAR,
                0, MPI_COMM_WORLD);

    // Step 7: Root prints the result
    if (rank == 0) {
        recv_buf[total_len] = '\0';
        printf("Output word: %s\n", recv_buf);
        free(recv_buf);
        free(recv_counts);
        free(displs);
    }

    free(local_str);
    MPI_Finalize();
    return 0;
}

