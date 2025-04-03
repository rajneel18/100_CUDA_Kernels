# Hebbian Learning in CUDA C

## **Introduction**
Hebbian learning is a fundamental neural learning rule where the weights between neurons are updated based on their activation correlation. The weight update follows:

\[
\Delta w_{ij} = \eta x_i y_j
\]

where:
- \( \eta \) is the learning rate,
- \( x_i \) is the input activation,
- \( y_j \) is the output activation.

CUDA allows efficient parallel updates of weights due to the independent nature of these calculations.

---

## **CUDA Implementation Plan**
1. **Define Input and Output Neuron Activations**: Store them in device memory.
2. **Parallel Weight Update Kernel**: Each thread computes one weight update.
3. **Copy Updated Weights Back to Host**: Retrieve the results from GPU.
