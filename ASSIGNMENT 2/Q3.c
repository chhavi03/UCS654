#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void odd_even_sort(int *array, int n, int rank, int size) {
    int temp;
    for (int phase = 0; phase < n; phase++) {
      
        if (phase % 2 == 0) {
            if (rank % 2 == 0 && rank < size - 1) {
             
                MPI_Send(&array[n - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (array[n - 1] > temp) {
                    
                    int temp_val = array[n - 1];
                    array[n - 1] = temp;
                    temp = temp_val;
                }
            } else if (rank % 2 == 1) {
                
                MPI_Recv(&temp, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&array[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                if (array[0] > temp) {
                  
                    int temp_val = array[0];
                    array[0] = temp;
                    temp = temp_val;
                }
            }
        } else {
            if (rank % 2 == 1 && rank < size - 1) {
            
                MPI_Send(&array[n - 1], 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&temp, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (array[n - 1] > temp) {
                   
                    int temp_val = array[n - 1];
                    array[n - 1] = temp;
                    temp = temp_val;
                }
            } else if (rank % 2 == 0) {
                
                MPI_Recv(&temp, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&array[0], 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
                if (array[0] > temp) {
                 
                    int temp_val = array[0];
                    array[0] = temp;
                    temp = temp_val;
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 16; 
    int *array = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        array = (int *)malloc(n * sizeof(int));
     
        for (int i = 0; i < n; i++) {
            array[i] = rand() % 100; 
        }
        printf("Unsorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    int *local_array = (int *)malloc(local_n * sizeof(int));

    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM