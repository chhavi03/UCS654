# Parallel Computing Experiments with MPI

This repository contains several parallel computing experiments implemented using MPI (Message Passing Interface). Each experiment demonstrates a different parallel computing technique.

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

## How to Compile and Run

Ensure you have MPI installed. You can compile and run the experiments using the following commands:

### Compilation
```sh
mpicc Q1.c -o Q1
mpicc Q2.c -o Q2
mpicc Q3.c -o Q3
mpicc Q4.c -o Q4
mpicc Q5.c -o Q5
mpicc Q6.c -o Q6
mpicc Q7.c -o Q7
mpicc Q8.c -o Q8
```

### Execution
```sh
mpirun -np <num_processes> ./Q1
mpirun -np <num_processes> ./Q2
mpirun -np <num_processes> ./Q3
mpirun -np <num_processes> ./Q4
mpirun -np <num_processes> ./Q5
mpirun -np <num_processes> ./Q6
mpirun -np <num_processes> ./Q7
mpirun -np <num_processes> ./Q8
```

Replace `<num_processes>` with the desired number of processes.

## Requirements
- MPI Library (e.g., OpenMPI, MPICH)
- C Compiler (e.g., GCC)

## License
This project is licensed under the MIT License.

