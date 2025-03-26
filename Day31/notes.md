# Gaussian Function (CUDA Notes)

## Definition

The Gaussian function (or Normal distribution's probability density function) is defined as:

### Key Concepts

* **Element-wise Operation:** The Gaussian function is typically applied to each element of an input array independently, making it suitable for parallel processing on a GPU.
* **Numerical Stability:** The `exp()` function can lead to numerical instability. Careful handling is required to prevent overflow or underflow.
