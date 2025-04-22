# Token Merging (ToMe) in CUDA

## Overview

**Token Merging (ToMe)** is a technique to reduce computational cost in transformers by merging similar token embeddings. This is useful for faster inference in LLMs without significantly affecting accuracy.

---

## Why Token Merging?

- Self-attention has quadratic time complexity: O(N²).
- ToMe merges semantically similar tokens (using cosine similarity).
- Reduces token count → faster inference with little accuracy loss.

---

## Step-by-Step Process

### 1. Cosine Similarity Computation



### 2. Merging Tokens

- Select token pairs with highest similarity.
- Merge by averaging:

