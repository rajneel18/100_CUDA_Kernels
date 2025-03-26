# Binary Cross Entropy (BCE) (CUDA Notes)

## Definition

Binary Cross Entropy (BCE) is a loss function used for binary classification tasks. It measures the difference between predicted probabilities and true binary labels.

Formula:


Where:

* $y_i$ is the true binary label (0 or 1).
* $\hat{y}_i$ is the predicted probability (between 0 and 1).
* $N$ is the number of samples.

### Key Concepts

* **Element-wise Operations:** The loss is calculated element-wise for each sample, making it suitable for parallel processing.
* **Numerical Stability:** The `log()` function can lead to numerical instability. Clipping or using a small epsilon is necessary.
* **Reduction:** The summation is a reduction operation.
