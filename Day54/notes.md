# Mandelbrot Set (CUDA Notes)

## Concept

The Mandelbrot set is a fractal defined by a simple iterative complex number equation. CUDA can be used to efficiently calculate and visualize this set.

## Mathematical Formulation

The Mandelbrot set is defined by the following iterative equation:

$$
z_{n+1} = z_n^2 + c
$$

Where:

* $z$ and $c$ are complex numbers.
* $z_0 = 0$.
* $c$ is a point in the complex plane.

If the absolute value of $z$ remains bounded (typically below 2) after a certain number of iterations, the point $c$ is considered to be part of the Mandelbrot set.

## CUDA Implementation

### Key Considerations

* **Complex Number Operations:** Implement complex number arithmetic (addition, multiplication, absolute value).
* **Iteration:** Perform the iterative calculation for each pixel in parallel.
* **Color Mapping:** Map the number of iterations to a color for visualization.
* **Parallelism:** Each pixel's calculation is independent, making it highly parallelizable.

