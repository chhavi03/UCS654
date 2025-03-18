#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void prefix_sum(int *input, int *output, int n, int rank, int size) {
    int local_sum = 0;
    for (int i = 0; i < n; i++) {
        local_sum += input[i];
        output[i] = local_sum;
    }
    
    int *recv_buf = (int *)malloc(size * sizeof(int));
    MPI_Gather(&local_sum, 1, MPI_INT, recv_buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int total_sum = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            int temp = recv_buf[i];
            recv_buf[i] = total_sum;
            total_sum += temp;
        }
    }
    
    MPI_Scatter(recv_buf, 1, MPI_INT, &total_sum, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    for (int i = 0; i < n; i++) {
        output[i] += total_sum;
    }
    
    free(recv_buf);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int n = 8;
    int *input = NULL;
    int *output = (int *)malloc(n * sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        input = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) {
            input[i] = i + 1;
        }
    }

    int local_n = n / size;
    int *local_input = (int *)malloc(local_n * sizeof(int));
    int *local_output = (int *)malloc(local_n * sizeof(int));

    MPI_Scatter(input, local_n, MPI_INT, local_input, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    
    prefix_sum(local_input, local_output, local_n, rank, size);
    
    MPI_Gather(local_output, local_n, MPI_INT, output, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Prefix Sum: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
        free(input);
    }

    free(local_input);
    free(local_output);
    free(output);
    MPI_Finalize();
    return 0;
}