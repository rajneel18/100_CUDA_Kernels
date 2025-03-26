# ReLU (Rectified Linear Unit) (CUDA Notes)

## Definition

The ReLU activation function is defined as:

$$
ReLU(x) = max(0, x)
$$


### Key Concepts

* **Element-wise Operation:** ReLU is applied independently to each element of an input array, making it highly parallelizable on a GPU.
* **Simplicity:** The calculation is very simple, leading to efficient CUDA implementations.
