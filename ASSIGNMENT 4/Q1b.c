#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumFirstN(int n, int *result) {
    if (threadIdx.x == 0) {
        *result = (n * (n + 1)) / 2;
    }
}

int main() {
    int n = 1024;
    int *d_result;
    int h_result;

    cudaMalloc((void **)&d_result, sizeof(int));

    sumFirstN<<<1, 1>>>(n, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum of the first %d integers: %d\n", n, h_result);

    cudaFree(d_result);

    return 0;
}