#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX_LEN 100

void toggle_case(char *word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (isupper(word[i]))
            word[i] = tolower(word[i]);
        else if (islower(word[i]))
            word[i] = toupper(word[i]);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    char word[MAX_LEN];

    MPI_Init(&argc, &argv);              // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (size < 2) {
        if (rank == 0)
            printf("This program requires at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // Sender Process
        strcpy(word, "HelloMPI"); // Initial word

        printf("Process 0 sending word: %s\n", word);
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD); // Synchronous send to Process 1

        MPI_Recv(word, MAX_LEN, MPI_CHAR, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive toggled word
        printf("Process 0 received toggled word: %s\n", word);

    } else if (rank == 1) {
        // Receiver Process
        MPI_Recv(word, MAX_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive word from Process 0

        printf("Process 1 received word: %s\n", word);
        toggle_case(word); // Toggle case

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD); // Synchronous send back to Process 0
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}


