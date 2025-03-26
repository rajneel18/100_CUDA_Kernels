# SwiGLU (Swish-Gated Linear Unit) (CUDA Notes)

## Concept

SwiGLU is an activation function that combines the Swish activation with a gated linear unit. It has shown promising results in transformer models.

## Mathematical Formulation

SwiGLU is defined as:

$$
SwiGLU(x, W, V) = Swish(xW) \otimes (xV)
$$

Where:

* $x$ is the input tensor.
* $W$ and $V$ are weight matrices.
* $Swish(x) = x \cdot \sigma(x)$, where $\sigma(x)$ is the sigmoid function.
* $\otimes$ denotes element-wise multiplication.

## CUDA Implementation

### Key Considerations

* **Matrix Multiplication:** The computation involves matrix multiplications ($xW$ and $xV$), which can be efficiently performed using cuBLAS.
* **Swish Activation:** The Swish activation is applied element-wise.
* **Element-wise Multiplication:** The final step is an element-wise multiplication.
