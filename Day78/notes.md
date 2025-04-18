# Sparse Attention – Notes

## Overview

Sparse Attention is an optimized version of the traditional self-attention mechanism, particularly useful for **long sequences**. It reduces the **quadratic complexity** of full attention (O(n²)) to **linear or sub-quadratic** by only computing attention for a sparse subset of token pairs.

---

## Why Sparse Attention?

Standard attention computes the attention score for every pair of tokens:

