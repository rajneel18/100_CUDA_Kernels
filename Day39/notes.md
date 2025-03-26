Markdown

# cuBLAS Matrix Multiplication (matmul) (Markdown Notes)

## Concept

cuBLAS provides highly optimized matrix multiplication routines for NVIDIA GPUs. This is a core operation in many applications, especially deep learning.

## Mathematical Formulation

Given matrices `A` (m x k), `B` (k x n), and `C` (m x n), the matrix multiplication operation is:

C = α * A * B + β * C


Where:

* `A`, `B`, and `C` are matrices.
* `α` and `β` are scalar multipliers.

## cuBLAS API

* `cublasSgemm()`: Performs single-precision matrix multiplication.
* `cublasDgemm()`: Performs double-precision matrix multiplication.
* `cublasCgemm()`: Performs single-precision complex matrix multiplication.
* `cublasZgemm()`: Performs double-precision complex matrix multiplication.
* `cublasHgemm()`: Performs half-precision matrix multiplication.
* `cublasGemmEx()`: A more flexible version that allows for mixed precision.
