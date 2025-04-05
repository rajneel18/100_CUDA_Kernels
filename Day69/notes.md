#  Vector Similarity Search in CUDA C (for RAG)

## Idea:
Compute cosine similarity between a **query vector** and **N document vectors**, then return the top-K most similar.

---

## Parameters
- `N`: number of document vectors
- `D`: embedding dimension
- `K`: top-k matches

---

## CUDA Parallelism
- Each thread computes cosine similarity for one document.
- Final results sorted on CPU for simplicity (can be done in CUDA too).
