#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000

int main(int argc, char *argv[]) {
    int rank, size;
    double *a = NULL, *b = NULL;
    double local_sum = 0.0, global_sum = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        a = (double *)malloc(N * sizeof(double));
        b = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            a[i] = rand() % 10;
            b[i] = rand() % 10;
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = N / size;
    double *local_a = (double *)malloc(local_n * sizeof(double));
    double *local_b = (double *)malloc(local_n * sizeof(double));

    MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_b, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_sum += local_a[i] * local_b[i];
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Dot Product: %f\n", global_sum);
        free(a);
        free(b);
    }

    free(local_a);
    free(local_b);
    MPI_Finalize();
    return 0;
}