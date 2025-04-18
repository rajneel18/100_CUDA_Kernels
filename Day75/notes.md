#  Embedding Lookup in CUDA

##  What is Embedding Lookup?

Embedding lookup is a process where **each token (word or ID)** is mapped to a **fixed-size vector** using a pre-trained or learnable matrix called the **embedding matrix**.

- **Input**: Sequence of token IDs → `[4, 2, 7]`
- **Embedding Matrix**: `Vocab_Size x Embedding_Dim`
- **Output**: Matrix of shape `Sequence_Length x Embedding_Dim`

---

##  Why Parallelize in CUDA?

- Token embeddings are independent of each other.
- Each lookup can be done in parallel across CUDA threads.
- Useful for **speeding up input preparation in transformers**.

---

##  CUDA Logic for Embedding Lookup

###  Inputs:
- `token_ids[]`: Sequence of input token IDs.
- `embedding_matrix[][]`: Predefined or learnable matrix of size `[vocab_size][embedding_dim]`

###  Output:
- `output_embeddings[][]`: Retrieved embeddings for each token.
