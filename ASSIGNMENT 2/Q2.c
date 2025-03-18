#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 70 

void multiply_matrices(int a[N][N], int b[N][N], int c[N][N], int rows) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0; 
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int a[N][N], b[N][N], c[N][N];
    int rows_per_process, start_row, end_row;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = rand() % 10; 
                b[i][j] = rand() % 10; 
            }
        }
    }

    MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    rows_per_process = N / size;
    start_row = rank * rows_per_process;
    end_row = (rank + 1) * rows_per_process;

    MPI_Scatter(a, rows_per_process * N, MPI_INT, a + start_row * N, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime(); 
    multiply_matrices(a + start_row * N, b, c + start_row * N, rows_per_process); // Perform matrix multiplication
    double run_time = MPI_Wtime() - start_time; 
    MPI_Gather(c + start_row * N, rows_per_process * N, MPI_INT, c, rows_per_process * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI execution time: %f seconds);