# Dynamic Time Warping (DTW) in CUDA (Notes)

## Concept

Dynamic Time Warping (DTW) is an algorithm for measuring similarity between two temporal sequences that may vary in speed. It finds the optimal alignment between the sequences by warping the time axis.

## Mathematical Formulation

Given two time series `X` and `Y` of lengths `m` and `n` respectively, DTW computes the optimal alignment path by minimizing the cumulative distance.

1.  **Distance Matrix:**
    * Create a distance matrix `D` of size `m x n` where `D(i, j) = dist(X[i], Y[j])`.
2.  **Cumulative Cost Matrix:**
    * Create a cumulative cost matrix `C` of size `(m+1) x (n+1)`.
    * Initialize `C(0, 0) = 0`, `C(i, 0) = infinity`, and `C(0, j) = infinity`.
    * Fill `C` using the recurrence relation:

    ```
    C(i, j) = D(i-1, j-1) + min(C(i-1, j), C(i, j-1), C(i-1, j-1))
    ```

3.  **DTW Distance:**
    * The DTW distance is `C(m, n)`.

## CUDA Implementation

### Key Considerations

* **Distance Matrix Calculation:** The distance matrix `D` can be calculated in parallel.
* **Cumulative Cost Matrix Calculation:** The recurrence relation for `C` has dependencies, requiring careful parallelization.
* **Memory Access:** Efficient memory access is crucial for performance.
* **Boundary Conditions:** Handling boundary conditions (first row and column of `C`) is essential.
