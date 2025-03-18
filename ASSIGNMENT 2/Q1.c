#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int rank, size, n, i, count = 0;
    double x, y, pi;
    long long total_points = 1000000; 
    long long points_per_process;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    points_per_process = total_points / size;

    srand(time(NULL) + rank); 

    for (i = 0; i < points_per_process; i++) {
        x = (double)rand() / RAND_MAX; 
        y = (double)rand() / RAND_MAX; 

        if (x * x + y * y <= 1.0) {
            count++;
        }
    }

    long long total_count;
    MPI_Reduce(&count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        pi = (double)total_count / total_points * 4.0; 
        printf("Estimated value of Pi: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}