# CUDA Project: Parallel Tokenization (Token-to-ID Mapping)

## Objective
Accelerate the token-to-ID mapping process in natural language preprocessing using CUDA. Each CUDA thread performs a string comparison against a fixed vocabulary and outputs the corresponding token ID.

---

## Concept

- Given: Pre-tokenized input sequences (words split by spaces).
- Goal: Convert each word to its integer ID from a fixed vocabulary in parallel.
- Method:
  - Store vocabulary words and IDs in constant memory.
  - Launch one thread per token to perform vocabulary lookup using string matching.
  - Output the ID (or `0` for unknown token).

---

## CUDA Features Used

- **Constant Memory**: For fast access to vocabulary words and IDs.
- **Thread Parallelism**: Each thread handles one token-to-ID conversion.
- **GPU Memory Management**: Host-to-device and device-to-host transfers.
- **Basic String Matching**: Parallel matching without hashing.
