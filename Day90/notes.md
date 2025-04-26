# Fused Linear + Softmax CUDA Kernel

## Introduction

In deep learning models, especially large ones like Transformers, a common pattern is:
- Linear Transformation (Matrix Multiplication)
- Bias Addition
- Softmax Activation

Instead of implementing these in separate kernels (leading to redundant memory loads), **Fused Linear + Softmax** combines them into a single efficient CUDA kernel.

This greatly improves **memory bandwidth efficiency** and **reduces kernel launch overhead**.

## Mathematical Formulation

Given:
- Input vector \( x \in \mathbb{R}^{d_{\text{input}}} \)
- Weight matrix \( W \in \mathbb{R}^{d_{\text{input}} \times d_{\text{output}}} \)
- Bias vector \( b \in \mathbb{R}^{d_{\text{output}}} \)

The fused operation computes:

1. Linear Transformation:

\[
z = xW + b
\]

2. Softmax Activation:

\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

where \( z_i \) is the \( i \)-th component of the vector \( z \).

## CUDA Kernel Workflow

1. **Load input and compute Linear layer**:
   - For each output dimension, accumulate dot products between input vector and corresponding weight column.
   - Add bias.

2. **Compute maximum value** (for numerical stability in softmax):
   - Find the maximum \( z_{\text{max}} \) across all outputs.

3. **Exponentiate and Sum**:
   - Subtract \( z_{\text{max}} \) from each logit.
   - Apply the exponential function.
   - Sum the exponentiated values.

4. **Normalize to get Softmax output**:
   - Divide each exponentiated logit by the total sum.

## CUDA Implementation Details

- **Shared Memory**:
  - Used to store intermediate logits during computation to reduce global memory accesses.

- **Atomic Operations**:
  - Used for computing the maximum logit and the sum of exponentials across threads.

- **Thread Distribution**:
  - Threads within a block are responsible for different output dimensions.

- **Batch Processing**:
  - Each block processes one batch item independently.

## Advantages

- **Better Memory Efficiency**:
  - Input is read only once instead of separately for Linear and Softmax.

- **Fewer Kernel Launches**:
  - Reduces overhead and improves latency, especially important in large-scale inference.

- **Higher Throughput**:
  - Parallelizes across output dimensions efficiently.

## Potential Optimizations

- Use **half-precision (FP16)** with Tensor Cores for faster computation.
- Fuse additional layers like **Dropout** or **Residual connections** into the same kernel.
- Use **warp-level primitives** (`__shfl_sync`) instead of atomic operations for faster reductions.
- Apply **Mixed Precision Training** to balance speed and accuracy.

## Example Application

This kernel is crucial in:
- Transformer Attention heads
- Final classification heads
- Large Language Models (LLMs) during inference

