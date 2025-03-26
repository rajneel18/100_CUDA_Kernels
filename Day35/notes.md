# Binary Cross Entropy with Softmax (BCE with Softmax) (CUDA Notes)

## Concept

BCE with Softmax combines the Softmax activation function with the Binary Cross Entropy loss. This is often used in multi-label classification problems where each label is independent.

## Mathematical Formulation

1.  **Softmax Activation:**

    $$
    \hat{y}_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
    $$

    where $x_i$ are the raw outputs (logits) and $\hat{y}_i$ are the predicted probabilities.

2.  **Binary Cross Entropy Loss:**

    $$
    BCE(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
    $$

    where $y_i$ are the true binary labels.
