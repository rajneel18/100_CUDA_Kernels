
## Definition

**Total Variation Distance** is a measure of how different two probability distributions \( P \) and \( Q \) are.

The mathematical formula is:

\[
\text{TV}(P, Q) = \frac{1}{2} \sum_i |P_i - Q_i|
\]

This value ranges from 0 (identical distributions) to 1 (completely disjoint distributions).

---

##  Intuition

- It measures the **maximum possible difference** between two distributions over the same probability space.
- Often used in **generative modeling** to evaluate the quality of outputs.
- For discrete distributions, it's just the half of the L1-norm difference.

---

##  CUDA Implementation Overview

1. Launch a CUDA kernel to calculate \( |P_i - Q_i| \times \frac{1}{2} \) for each index \( i \).
2. Use **shared memory reduction** to sum the values within each block.
3. Use **`atomicAdd`** to accumulate partial sums into the final result.
