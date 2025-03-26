## CUDA Layer Normalization (LayerNorm) (Short Notes)

**Concept:** Implementing Layer Normalization efficiently on GPUs using CUDA.

**Layer Normalization:**

* Normalizes the activations within each sample across its features.
* Formula: `y = (x - mean(x)) / sqrt(variance(x) + epsilon) * gamma + beta`.

**CUDA LayerNorm Forward Pass:**

1.  **Mean Calculation:**
    * Calculate the mean of the features for each sample.
    * CUDA: Reduction kernel (e.g., using shared memory and atomic operations) to calculate the mean.
2.  **Variance Calculation:**
    * Calculate the variance of the features for each sample.
    * CUDA: Reduction kernel to calculate the variance.
3.  **Normalization:**
    * Subtract the mean and divide by the standard deviation (sqrt of variance + epsilon).
    * CUDA: Element-wise kernel.
4.  **Scaling and Shifting:**
    * Multiply by gamma (scale) and add beta (shift).
    * CUDA: Element-wise kernel.

