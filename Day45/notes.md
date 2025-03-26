# 3-Body Problem 
## Concept

The 3-body problem involves simulating the gravitational interactions of three bodies in space. CUDA can significantly accelerate this simulation by parallelizing the force calculations and position updates.

## Mathematical Formulation

The gravitational force between two bodies is given by:

$$
F_{ij} = G \frac{m_i m_j}{r_{ij}^2} \hat{r}_{ij}
$$

Where:

* $F_{ij}$ is the force exerted on body $i$ by body $j$.
* $G$ is the gravitational constant.
* $m_i$ and $m_j$ are the masses of the bodies.
* $r_{ij}$ is the distance between the bodies.
* $\hat{r}_{ij}$ is the unit vector pointing from body $j$ to body $i$.

The acceleration of each body is:

$$
a_i = \frac{F_i}{m_i}
$$

Where $F_i$ is the net force acting on body $i$.

The position and velocity updates are typically done using a numerical integration method (e.g., Euler's method, Verlet integration).

## CUDA Implementation

### Key Considerations

* **Parallel Force Calculation:** The force calculations between each pair of bodies can be done in parallel.
* **Position and Velocity Updates:** The position and velocity updates can also be parallelized.
* **Memory Access:** Ensure coalesced memory access for optimal performance.
* **Numerical Stability:** The simulation can be sensitive to numerical errors.
