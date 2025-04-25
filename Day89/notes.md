# MROPE - Memory-Efficient Rotary Positional Embedding

## Introduction

Memory-efficient Rotary Positional Embeddings (MROPE) are an optimization over Rotary Positional Embeddings (RoPE), commonly used in transformer-based LLMs to encode positional information in token embeddings without requiring explicit position vectors.

## Motivation

Traditional RoPE techniques apply rotation matrices to queries and keys based on absolute positions. This can be memory- and compute-intensive, especially on long sequences. MROPE reduces memory overhead by:

- **Avoiding redundant computations**
- **Exploiting shared frequencies**
- **Computing embeddings on-the-fly**
