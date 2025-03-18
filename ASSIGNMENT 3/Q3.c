#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int is_prime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int max_value = 100; // Set the maximum value for prime testing
    int number;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (number = 2; number <= max_value; number++) {
            int source;
            MPI_Recv(&source, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_SOURCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&number, 1, MPI_INT, source, 0, MPI_COMM_WORLD);
        }
        for (int i = 1; i < size; i++) {
            int end_signal = 0;
            MPI_Send(&end_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        while (1) {
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (number == 0) break;
            if (is_prime(number)) {
                MPI_Send(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            } else {
                int non_prime = -number;
                MPI_Send(&non_prime, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Recv(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (number > 0) {
                printf("%d is a prime number.\n", number);
            } else {
                printf("%d is not a prime number.\n", -number);
            }
        }
    }

    MPI_Finalize();
    return 0;
}