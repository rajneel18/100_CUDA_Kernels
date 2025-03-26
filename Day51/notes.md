# Radix Sort in CUDA (Notes)

## Concept

Radix sort is a non-comparative sorting algorithm that sorts data with integer keys by grouping keys by the individual digits which share the same significant position and value. CUDA can accelerate radix sort by parallelizing the counting and prefix sum operations.

## Algorithm

1.  **Determine the number of digits (or bits):** Find the maximum element to determine the number of digits/bits to process.
2.  **Iterate through digits/bits:** Process each digit/bit from the least significant to the most significant.
3.  **Counting sort:** For each digit/bit:
    * Count the occurrences of each digit/bit value (0-9 for decimal, 0-1 for binary).
    * Compute the prefix sum of the counts.
4.  **Rearrange elements:** Use the prefix sum to determine the correct position of each element in the sorted output.

## CUDA Implementation

### Key Concepts

* **Parallel Counting:** CUDA can parallelize the counting of digit/bit occurrences.
* **Parallel Prefix Sum (Scan):** Efficiently compute the prefix sum using CUDA's parallel scan algorithms.
* **Global Memory Access:** Minimize global memory access for performance.
