# Parallel Computing Experiments with MPI and CUDA

This repository contains several parallel and GPU-based computing experiments implemented using MPI and CUDA. Each experiment demonstrates different techniques in high-performance computing.

## Experiments Overview

### Assignment 2

#### 1. Monte Carlo Method for Estimating Pi (`Q1.c`)
- Uses the Monte Carlo simulation to estimate the value of Pi.
- Each process generates random points and checks if they fall within a unit circle.
- Results are reduced using `MPI_Reduce` to compute the final Pi estimate.

#### 2. Parallel Matrix Multiplication (`Q2.c`)
- Implements matrix multiplication using MPI.
- The matrix is divided among multiple processes for parallel computation.
- Uses `MPI_Scatter` to distribute rows and `MPI_Gather` to collect the results.

#### 3. Parallel Odd-Even Sort (`Q3.c`)
- Implements parallel odd-even sorting.
- Each process sorts its local portion and communicates with neighboring processes.
- Uses `MPI_Send` and `MPI_Recv` to exchange boundary elements.

#### 4. Parallel Heat Distribution Simulation (`Q4.c`)
- Simulates heat distribution over a 2D grid.
- Uses iterative updates with MPI-based row communication.
- Demonstrates the use of ghost rows for proper boundary exchange.

#### 5. Parallel Sum Computation (`Q5.c`)
- Computes the sum of an array in parallel.
- Uses `MPI_Scatter` to distribute data and `MPI_Reduce` to sum up results.

#### 6. Parallel Dot Product (`Q6.c`)
- Computes the dot product of two vectors in parallel.
- Uses `MPI_Scatter` for data distribution and `MPI_Reduce` for final summation.

#### 7. Parallel Prefix Sum (`Q7.c`)
- Implements the prefix sum operation using MPI.
- Each process computes its local prefix sum, followed by an MPI-based adjustment.

#### 8. Parallel Matrix Transposition (`Q8.c`)
- Performs matrix transposition in parallel.
- Uses `MPI_Scatter` and `MPI_Gather` for efficient data movement.

### Assignment 3

#### 1. Parallel DAXPY Computation (`Q1.c`)
- Implements the DAXPY operation (`y = ax + y`) in parallel.
- Uses `MPI_Scatter` to distribute data and `MPI_Gather` to collect results.
- Compares parallel execution time with sequential execution.

#### 2. Parallel Numerical Integration (`Q2.c`)
- Uses the midpoint rule to approximate the integral of `4 / (1 + x^2)` for estimating Pi.
- Each process computes a portion of the integral.
- Uses `MPI_Reduce` to sum up partial results.

#### 3. Parallel Prime Number Detection (`Q3.c`)
- Distributes numbers among processes to check for primality.
- Uses `MPI_Send` and `MPI_Recv` for process coordination.
- Each process independently determines if a given number is prime.

---

### Assignment 4

#### 1a. Combined Operations: Sum, Factorials, and Squares (`Q1a.c`)
- A CUDA kernel that computes:
  - Sum of first N integers using a single thread.
  - Factorials of integers from 0 to N-1.
  - Squares of integers from 0 to N-1.
- Demonstrates multiple parallel computations in a single kernel using thread indices.

#### 1b. Sum of First N Integers (`Q1b.c`)
- A simple CUDA program to compute the sum of the first N integers using a single thread in a kernel.
- Emphasizes the use of device memory and host-device communication.

---

### Assignment 6

#### 2. GPU Square Root Computation with Output (`Q2.cpp`)
- A CUDA program that computes the square root of each element in an array.
- Outputs the square roots of each element to the console.
- Demonstrates basic kernel execution and host-device memory management.

#### 3. Timed Square Root Computation for Multiple Input Sizes (`Q3.c`)
- Measures the time taken by a CUDA kernel to compute the square roots of elements in arrays of varying sizes.
- Useful for performance analysis and understanding CUDA kernel execution timing.

---

## How to Compile and Run

### MPI Programs
```sh
mpicc Q1.c -o Q1
mpirun -np <num_processes> ./Q1
```

### CUDA Programs
```sh
nvcc Q1a.c -o Q1a
nvcc Q1b.c -o Q1b
nvcc Q2.cpp -o Q2
nvcc Q3.c -o Q3
./Q1a
./Q1b
./Q2
./Q3
```

Replace `<num_processes>` with the desired number of MPI processes.

---

## Requirements
- MPI Library (e.g., OpenMPI, MPICH)
- CUDA Toolkit and NVIDIA GPU
- C/C++ Compiler (e.g., GCC, nvcc)

---

## License
This project is licensed under the MIT License.
