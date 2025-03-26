# Linear Kernel (CUDA Notes)

## Concept

The Linear Kernel is a simple kernel function used in machine learning, particularly in Support Vector Machines (SVMs). It calculates the dot product between two vectors.

## Mathematical Formulation

Given two vectors `x` and `y`, the Linear Kernel is defined as:

$$
K(x, y) = x^T y = \sum_{i=1}^{n} x_i y_i
$$

Where:

* $x$ and $y$ are vectors of dimension $n$.
* $x^T$ is the transpose of vector $x$.

## CUDA Implementation

### Key Considerations

* **Dot Product Calculation:** The core operation is the dot product, which involves element-wise multiplication and summation.
* **Reduction:** The summation part of the dot product requires a reduction operation.
* **Memory Access:** Ensure coalesced memory access for optimal performance.
