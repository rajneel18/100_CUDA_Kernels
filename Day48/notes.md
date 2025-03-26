# Conjugate Gradient (CG) Method (CUDA Notes)

## Concept

The Conjugate Gradient (CG) method is an iterative algorithm for solving linear systems of equations:

$$
Ax = b
$$

Where:

* $A$ is a symmetric, positive-definite matrix.
* $x$ is the unknown vector.
* $b$ is the known vector.

CG is particularly effective for large, sparse systems.

## Algorithm

1.  **Initialization:**
    * Choose an initial guess $x_0$.
    * Compute the residual $r_0 = b - Ax_0$.
    * Set the initial search direction $p_0 = r_0$.

2.  **Iteration:** For $k = 0, 1, 2, ...$ until convergence:
    * Compute the step size $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$.
    * Update the solution $x_{k+1} = x_k + \alpha_k p_k$.
    * Update the residual $r_{k+1} = r_k - \alpha_k A p_k$.
    * Compute $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$.
    * Update the search direction $p_{k+1} = r_{k+1} + \beta_k p_k$.

## CUDA Implementation

### Key Operations

* **Matrix-Vector Multiplication (Ax):** This is often the most computationally intensive part and can be efficiently parallelized on the GPU.
* **Vector Dot Products (r^T r, p^T Ap):** These can be implemented using reduction operations.
* **Vector Additions and Scalar Multiplications:** These are element-wise operations that are easily parallelized.
