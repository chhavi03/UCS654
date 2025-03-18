#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100 

int main(int argc, char *argv[]) {
    int rank, size;
    int *array = NULL;
    int local_sum = 0;
    int global_sum = 0;

  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
       
        for (int i = 0; i < N; i++) {
            array[i] = rand() % 10; 
        }
        printf("Array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = N / size;

    int *local_array = (int *)malloc(local_n * sizeof(int));

    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_n; i++) {
        local_sum += local_array[i];
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total sum: %d\n", global_sum);
    }

    if (rank == 0) {
        free(array);
    }
    free(local_array);

    MPI_Finalize();
    return 0;
}