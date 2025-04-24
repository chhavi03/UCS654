#include <stdio.h>
#include <cuda_runtime.h>

__global__ void performTasks(int n, int *sumResult, int *factorialResult, int *squareResult) {
    int idx = threadIdx.x;

    if (idx == 0) {
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += i;
        }
        *sumResult = sum;
    }

    if (idx < n) {
        int factorial = 1;
        for (int i = 1; i <= idx; i++) {
            factorial *= i;
        }
        factorialResult[idx] = factorial;
    }

    if (idx < n) {
        squareResult[idx] = idx * idx;
    }
}

int main() {
    int n = 1024;
    int *d_sumResult, *d_factorialResult, *d_squareResult;
    int h_sumResult, h_factorialResult[n], h_squareResult[n];

    cudaMalloc((void **)&d_sumResult, sizeof(int));
    cudaMalloc((void **)&d_factorialResult, n * sizeof(int));
    cudaMalloc((void **)&d_squareResult, n * sizeof(int));

    performTasks<<<1, n>>>(n, d_sumResult, d_factorialResult, d_squareResult);

    cudaMemcpy(&h_sumResult, d_sumResult, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_factorialResult, d_factorialResult, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_squareResult, d_squareResult, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of first %d integers: %d\n", n, h_sumResult);
    printf("Factorials:\n");
    for (int i = 0; i < n; i++) {
        printf("%d! = %d\n", i, h_factorialResult[i]);
    }
    printf("Squares:\n");
    for (int i = 0; i < n; i++) {
        printf("%d^2 = %d\n", i, h_squareResult[i]);
    }

    cudaFree(d_sumResult);
    cudaFree(d_factorialResult);
    cudaFree(d_squareResult);

    return 0;
}