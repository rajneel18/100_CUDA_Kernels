## CUDA Convolution Backward Pass (Short Notes)

**Concept:** Computes gradients for convolution layers during backpropagation.

**Key Components:**

* **`d_input`:** Gradient w.r.t. input.
* **`d_weight`:** Gradient w.r.t. weights.
* **`d_output`:** Gradient w.r.t. output.

**CUDA Kernel Logic:**

1.  **Thread Mapping:**
    * Each thread calculates a portion of `d_input` or `d_weight`.

    * Efficient indexing is crucial for accessing relevant data.
2.  **Shared Memory:**
    * Utilize shared memory to reduce global memory access.
    * Load input/weight patches into shared memory for faster computations.
3.  **Gradient Calculation:**
    * `d_input` calculation involves convolving `d_output` with flipped weights.
    * `d_weight` calculation involves convolving input with `d_output`.
    * Use appropriate summation/reduction techniques.
4.  **Atomic Operations:**
    * For `d_weight` accumulation, atomic operations (e.g., `atomicAdd`) might be necessary to handle race conditions.
5.  **Optimization:**
    * Tiling and blocking strategies for memory access patterns.
    * Loop unrolling for improved performance.

