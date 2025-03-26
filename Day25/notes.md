## CUDA ReLU Forward and Backward Pass (Short Notes)

**ReLU (Rectified Linear Unit):**

* A simple and effective activation function: `f(x) = max(0, x)`.

**CUDA ReLU Forward Pass:**

* **Concept:** Apply the ReLU function element-wise to the input.
* **Kernel Logic:**
    * Each thread processes one element of the input.
    * If the input element is greater than 0, output the element.
    * Otherwise, output 0.
