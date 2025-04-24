#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void computeSquareRoot(float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = sqrtf(A[i]); 
    }
}
int main() {
    int n = 1024; 
    size_t size = n * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
  
    for (int i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i); 
    }
    
    float *d_A, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_C, size);
   
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    computeSquareRoot<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, n);
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) {
        std::cout << "sqrt(" << h_A[i] << ") = " << h_C[i] << std::endl;
    }
    
    cudaFree(d_A);
    cudaFree(d_C);
    
    free(h_A);
    