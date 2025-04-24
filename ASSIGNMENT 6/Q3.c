#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void computeSquareRoot(float *A, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = sqrtf(A[idx]);
    }
}

void runTest(int N) {
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
    }

    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    computeSquareRoot<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for %d elements: %f ms\n", N, milliseconds);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
}

int main() {
    int sizes[] = {50000, 500000, 5000000, 50000000};
    for (int i = 0; i < 4; i++) {
        runTest(sizes[i]);
    }
    return 0;
}