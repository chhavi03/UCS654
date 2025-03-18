#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N (1 << 16)

void daxpy(double a, double *x, double *y, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a * x[i] + y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double a = 2.0;
    double *x = NULL;
    double *y = NULL;
    double *result = NULL;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        x = (double *)malloc(N * sizeof(double));
        y = (double *)malloc(N * sizeof(double));
        result = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            x[i] = i + 1.0;
            y[i] = 2.0 * i;
        }
    }

    double *local_x = (double *)malloc(N / size * sizeof(double));
    double *local_y = (double *)malloc(N / size * sizeof(double));
    double *local_result = (double *)malloc(N / size * sizeof(double));

    MPI_Bcast(y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(x, N / size, MPI_DOUBLE, local_x, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start_time = MPI_Wtime();
    daxpy(a, local_x, y, local_result, N / size);
    end_time = MPI_Wtime();

    MPI_Gather(local_result, N / size, MPI_DOUBLE, result, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI DAXPY execution time: %f seconds\n", end_time - start_time);
        double start_time_seq = MPI_Wtime();
        daxpy(a, x, y, result, N);
        double end_time_seq = MPI_Wtime();
        printf("Sequential DAXPY execution time: %f seconds\n", end_time_seq - start_time_seq);
        free(x);
        free(y);
        free(result);
    }

    free(local_x);
    free(local_y);
    free(local_result);
    MPI_Finalize();
    return 0;
}