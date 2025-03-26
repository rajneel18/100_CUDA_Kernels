# cuBLAS Vector Addition (Markdown Notes)

## Concept

cuBLAS (CUDA Basic Linear Algebra Subroutines) provides optimized implementations of common linear algebra operations on NVIDIA GPUs. Vector addition is a fundamental operation.

## Mathematical Formulation

Given two vectors `x` and `y` of size `n`, the vector addition operation is:
y = α * x + y


Where:

* `x` and `y` are input vectors.
* `α` is a scalar multiplier

## cuBLAS API

* `cublasSaxpy()`: Performs single-precision vector addition.
* `cublasDaxpy()`: Performs double-precision vector addition.
* `cublasCaxpy()`: Performs single-precision complex vector addition.
* `cublasZaxpy()`: Performs double-precision complex vector addition.
