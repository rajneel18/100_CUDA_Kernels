
# Triplet Loss in CUDA C

## Overview

Triplet loss is commonly used in metric learning to ensure that an anchor is closer to a positive (similar) sample than a negative (dissimilar) one by a margin.

Given:
- Anchor vector \( a \)
- Positive vector \( p \)
- Negative vector \( n \)
- Margin \( \alpha \)

The loss for each triplet is:

\[
L = \max(0, ||a - p||^2 - ||a - n||^2 + \alpha)
\]

Where:
- \( ||a - p||^2 \): Squared distance between anchor and positive
- \( ||a - n||^2 \): Squared distance between anchor and negative
- \( \alpha \): Margin that separates positives from negatives

## CUDA Implementation Details

### Device Function

Calculates squared Euclidean distance between two vectors:

```c
__device__ float euclidean_distance_squared(const float* a, const float* b, int dim);
```

### Kernel

Launches one thread per triplet:

```c
__global__ void triplet_loss_kernel(
    const float* anchor, const float* positive, const float* negative,
    float* losses, int batch_size, int dim, float margin);
```

Each thread:
- Computes \( d_{ap} = ||a - p||^2 \)
- Computes \( d_{an} = ||a - n||^2 \)
- Computes loss \( \max(d_{ap} - d_{an} + \alpha, 0) \)

## Sample Output

```
Triplet losses:
Sample 0: 0.120
Sample 1: 0.000
Sample 2: 0.003
Sample 3: 0.000
```

## Applications

- Face recognition (FaceNet)
- Few-shot learning
- Metric learning for retrieval

## Advantages

- Directly enforces a margin between similar and dissimilar pairs.
- More informative than pairwise (contrastive) loss.
