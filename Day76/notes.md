# Cosine Similarity in CUDA

## What is Cosine Similarity?

Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space. It is defined as the cosine of the angle between the vectors, and is commonly used in NLP, embeddings, and information retrieval.

## CUDA Parallelism Use Case

- Compute cosine similarities between multiple vector pairs (e.g., comparing a query against a large database)
- Speed up similarity search in applications such as document matching or feature comparison

## Kernel Strategy

- Each thread block handles one vector pair
- Shared memory used to accumulate partial sums
- Final result reduced and written to global memory

