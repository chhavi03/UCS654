#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

void transpose(int *matrix, int *transposed, int n, int rank, int size) {
    int local_n = n / size;
    int *local_matrix = (int *)malloc(local_n * n * sizeof(int));
    int *local_transposed = (int *)malloc(local_n * n * sizeof(int));

    MPI_Scatter(matrix, local_n * n, MPI_INT, local_matrix, local_n * n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < n; j++) {
            local_transposed[j * local_n + i] = local_matrix[i * n + j];
        }
    }

    MPI_Gather(local_transposed, local_n * n, MPI_INT, transposed, local_n * n, MPI_INT, 0, MPI_COMM_WORLD);

    free(local_matrix);
    free(local_transposed);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *matrix = NULL;
    int *transposed = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        matrix = (int *)malloc(N * N * sizeof(int));
        transposed = (int *)malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = i * N + j + 1;
            }
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    transpose(matrix, transposed, N, rank, size);

    if (rank == 0) {
        printf("Original Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[i * N + j]);
            }
            printf("\n");
        }
        printf("Transposed Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", transposed[i * N + j]);
            }
            printf("\n");
        }
        free(matrix);
        free(transposed);
    }

    MPI_Finalize();
    return 0;
}