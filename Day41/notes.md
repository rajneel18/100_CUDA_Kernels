# Gradient Descent vs. Mirror Descent (CUDA Notes)

## Concept

Gradient Descent (GD) and Mirror Descent (MD) are optimization algorithms used to minimize a function. CUDA can accelerate these algorithms by performing the computationally intensive parts of the update step on the GPU.

## Gradient Descent (GD)

### Definition

GD updates parameters in the negative direction of the function's gradient.

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

Where:

* $x_t$ is the parameter vector at iteration $t$.
* $\eta$ is the learning rate.
* $\nabla f(x_t)$ is the gradient of the function $f$ at $x_t$.

### CUDA Implementation

* The gradient calculation $\nabla f(x_t)$ is the primary computation that can be parallelized on the GPU.
* Element-wise operations (vector subtraction, scalar multiplication) are also suitable for CUDA.
