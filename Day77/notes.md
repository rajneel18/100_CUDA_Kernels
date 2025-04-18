# HNSW-KNN in CUDA(Hierarchical Navigable Small World)

## Overview

This CUDA implementation performs **Approximate Nearest Neighbor Search** using a simplified version of **HNSW (Hierarchical Navigable Small World)** graphs. Instead of exhaustive comparisons, it navigates a sparse graph to quickly find close neighbors, useful for high-dimensional embedding retrieval.

---

## What is HNSW?

- **HNSW** is a graph-based nearest neighbor search algorithm.
- Each data point is a **node** in a graph.
- Nodes are connected via **edges** to other "neighbor" nodes.
- Search starts from an entry point and traverses the graph using **greedy search**, only evaluating a few neighbors at each step.

---

## CUDA Implementation Strategy

- **Input**: Matrix of vectors (N x D), query vector (1 x D)
- **Output**: Top K nearest neighbors (approximate)

### Parallelism:
- **Block-level parallelism** for multiple query vectors
- **Thread-level parallelism** for distance calculations within a candidate pool
