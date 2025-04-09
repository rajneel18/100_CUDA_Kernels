# CUDA C - Attention Score Matrix Computation

## Objective
Compute the **Attention Score Matrix** used in Transformer models by performing the scaled dot-product between **Query (Q)** and **Key (K)** matrices in parallel using CUDA.

---

## Formula

```
Attention_Score[i][j] = (Q[i] · K[j]) / sqrt(D)
```

- `Q[i]` = i-th query vector  
- `K[j]` = j-th key vector  
- `D` = Embedding dimension (used for scaling)


This matrix `S` has shape `[seq_len, seq_len]` and contains the raw attention scores.

## Why Parallelize in CUDA?

- Attention matrix computation involves:
  - Multiple dot products between vectors (Q rows × K rows)
  - Scaling by `1 / √d_k`
- These are independent operations → ideal for GPU parallelism!
