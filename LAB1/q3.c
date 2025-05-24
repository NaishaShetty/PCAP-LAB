#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    float num1, num2, result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there are exactly 5 processes
    if (size != 5) {
        if (rank == 0) {
            printf("This program requires exactly 5 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        // Input two numbers
        printf("Enter two numbers: ");
        scanf("%f %f", &num1, &num2);

        // Send numbers to all other processes
        for (int i = 1; i < 5; i++) {
            MPI_Send(&num1, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&num2, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

        // Receive and display results
        char* operations[] = {"Addition", "Subtraction", "Multiplication", "Division"};
        for (int i = 1; i < 5; i++) {
            MPI_Recv(&result, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s result from process %d: %.2f\n", operations[i - 1], i, result);
        }
    } else {
        // Receive numbers
        MPI_Recv(&num1, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&num2, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform operation based on rank
        switch (rank) {
            case 1:
                result = num1 + num2;
                break;
            case 2:
                result = num1 - num2;
                break;
            case 3:
                result = num1 * num2;
                break;
            case 4:
                if (num2 != 0)
                    result = num1 / num2;
                else
                    result = 0.0; // simple error handling
                break;
        }

        // Send result back to root
        MPI_Send(&result, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

