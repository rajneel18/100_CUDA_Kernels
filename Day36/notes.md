# Gaussian Function for Hidden Layers (HL) (CUDA Notes)

## Concept

Applying the Gaussian function as an activation function within hidden layers of a neural network in CUDA. While less common than ReLU or GELU, it can be useful in specific scenarios.

## Mathematical Formulation

Gaussian function (Probability Density Function - PDF):

$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

Where:

* $x$ is the input to the hidden layer neuron.
* $\mu$ is the mean (often set to 0).
* $\sigma$ is the standard deviation (often set to 1).

## CUDA Implementation

### Key Considerations

* **Element-wise Operation:** The Gaussian function is applied element-wise, making it suitable for CUDA's parallel processing.
* **Numerical Stability:** The `exp()` function can be numerically unstable.
* **Gradient Calculation:** The gradient of the Gaussian function is required for backpropagation.
