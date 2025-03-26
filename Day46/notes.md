# Wave Function Evolution in CUDA (Notes)

## Concept

Simulating the time evolution of a wave function, often governed by the Schrödinger equation, using CUDA for parallel computation.

## Mathematical Formulation

The time-dependent Schrödinger equation:

$$
i\hbar \frac{\partial \Psi}{\partial t} = \hat{H} \Psi
$$

Where:

* $\Psi(x, t)$ is the wave function.
* $\hat{H}$ is the Hamiltonian operator.
* $\hbar$ is the reduced Planck constant.

For numerical simulation, we often discretize the equation and use time-stepping methods (e.g., Euler method, Crank-Nicolson method).

## CUDA Implementation

### Key Considerations

* **Discretization:** Discretize the wave function and the Hamiltonian operator.
* **Time-Stepping:** Choose a suitable time-stepping method.
* **Complex Numbers:** Wave functions are often complex-valued.
* **Parallel Computation:** CUDA is well-suited for parallel computation of the update steps.
