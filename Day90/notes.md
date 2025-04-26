# Fused Linear + Softmax CUDA Kernel

## Introduction

In deep learning models, especially large ones like Transformers, a common pattern is:
- Linear Transformation (Matrix Multiplication)
- Bias Addition
- Softmax Activation

Instead of implementing these in separate kernels (leading to redundant memory loads), Fused Linear + Softmax combines them into a single efficient CUDA kernel.

This greatly improves memory bandwidth efficiency and reduces kernel launch overhead.

## Mathematical Formulation

Given:
- Input vector ( x ∈ ℝ^(d_input) )
- Weight matrix ( W ∈ ℝ^(d_input × d_output) )
- Bias vector ( b ∈ ℝ^(d_output) )

The fused operation computes:

1. Linear Transformation:

    z = xW + b

2. Softmax Activation:

    Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)

where ( zᵢ ) is the i-th component of the vector ( z ).

## CUDA Kernel Workflow

1. Load input and compute Linear layer:
   - For each output dimension, accumulate dot products between input vector and corresponding weight column.
   - Add bias.

2. Compute maximum value (for numerical stability in softmax):
   - Find the maximum z_max across all outputs.

3. Exponentiate and Sum:
   - Subtract z_max from each logit.
   - Apply the exponential function.
   - Sum the exponentiated values.

4. Normalize to get Softmax output:
   - Divide each exponentiated logit by the total sum.

