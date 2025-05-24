#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int M;
    float *full_data = NULL;
    float *recv_data;
    float local_sum = 0.0, local_avg = 0.0;
    float *all_avgs = NULL, total_avg = 0.0;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (rank == 0) {
        printf("Enter number of elements per process (M): ");
        scanf("%d", &M);

        full_data = (float *)malloc(M * size * sizeof(float));

        printf("Enter %d elements:\n", M * size);
        for (int i = 0; i < M * size; i++) {
            scanf("%f", &full_data[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate space for each process to receive M elements
    recv_data = (float *)malloc(M * sizeof(float));

    // Scatter the data: send M elements to each process
    MPI_Scatter(full_data, M, MPI_FLOAT, recv_data, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Each process computes local average
    for (int i = 0; i < M; i++) {
        local_sum += recv_data[i];
    }
    local_avg = local_sum / M;

    // Root gathers all local averages
    if (rank == 0) {
        all_avgs = (float *)malloc(size * sizeof(float));
    }

    MPI_Gather(&local_avg, 1, MPI_FLOAT, all_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            total_avg += all_avgs[i];
        }
        total_avg /= size;
        printf("Total average of all %d elements = %.2f\n", M * size, total_avg);

        free(full_data);
        free(all_avgs);
    }

    free(recv_data);
    MPI_Finalize(); // Finalize MPI
    return 0;
}

