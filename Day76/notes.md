# Cosine Similarity in CUDA

## What is Cosine Similarity?

Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space. It is defined as the cosine of the angle between the vectors, and is often used in high-dimensional positive spaces such as text data or embeddings.

### Mathematical Formula

\[
\text{cosine\_similarity} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \cdot \|\vec{B}\|}
\]

Where:
- \( \vec{A} \cdot \vec{B} = \sum_{i=1}^{D} A_i B_i \) is the dot product
- \( \|\vec{A}\| = \sqrt{\sum_{i=1}^{D} A_i^2} \) is the L2 norm (magnitude) of A
- \( \|\vec{B}\| = \sqrt{\sum_{i=1}^{D} B_i^2} \) is the L2 norm of B

## Why CUDA?

- Allows for computing similarity between many vector pairs in parallel
- Useful in applications like sentence similarity, nearest neighbors, clustering, and recommendation systems
- Enables high throughput when working with large vector databases

## CUDA Implementation Overview

### Inputs:
- Arrays `A` and `B` representing two vectors of size `D`
- Output: cosine similarity value

### CUDA Kernel Steps:
1. Compute dot product \( \vec{A} \cdot \vec{B} \)
2. Compute L2 norms \( \|\vec{A}\| \) and \( \|\vec{B}\| \)
3. Combine into final similarity measure using the cosine formula
