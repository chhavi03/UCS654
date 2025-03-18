#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100 
#define ITERATIONS 1000 
#define ALPHA 0.1 

void initialize_grid(double grid[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i][j] = 0.0; 
        }
    }

    grid[N / 2][N / 2] = 100.0; 
}

void print_grid(double grid[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", grid[i][j]);
        }
        printf("\n");
    }
}

void update_grid(double grid[N][N], double new_grid[N][N], int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < N - 1; j++) {
            new_grid[i][j] = grid[i][j] + ALPHA * (
                grid[i - 1][j] + grid[i + 1][j] + grid[i][j - 1] + grid[i][j + 1] - 4 * grid[i][j]
            );
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double grid[N][N], new_grid[N][N];
    double *local_grid, *local_new_grid;
    int rows_per_process, start_row, end_row;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    rows_per_process = N / size;
    local_grid = (double *)malloc((rows_per_process + 2) * N * sizeof(double)); // +2 for ghost rows
    local_new_grid = (double *)malloc((rows_per_process + 2) * N * sizeof(double));

    if (rank == 0) {
        initialize_grid(grid);
    }

    MPI_Scatter(grid, rows_per_process * N, MPI_DOUBLE, local_grid + N, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < N; j++) {
            local_grid[0 * N + j] = grid[0][j];
        }
    }
    if (rank == size - 1) {
        for (int j = 0; j < N; j++) {
            local_grid[(rows_per_process + 1) * N + j] = grid[N - 1][j]; 
        }
    }

    for (int iter = 0; iter < ITERATIONS; iter++) {
      
        update_grid(local_grid, local_new_grid, 1, rows_per_process + 1);

        if (rank > 0) {
            MPI_Send(local_grid + N, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD); 
            MPI_Recv(local_grid, N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive from top
        }
        if (rank < size - 1) {
            MPI_Send(local_grid + rows_per_process * N, N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD); //