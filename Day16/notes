# Matrix Factorization and ALS

## 1. Basic Matrix Factorization Model

### What is Matrix Factorization?

Matrix factorization is a popular approach for recommendation systems that aims to predict missing values in a user-item interaction matrix (ratings matrix).  
It decomposes the large user-item matrix into two smaller matrices:


These matrices capture the hidden relationships between users and items.

### Goal:  
Predict the unknown rating by computing the dot product of user factors and item factors:  




### Objective Function:  
The optimization goal is to minimize the following loss function:  


---

## 2. Alternating Least Squares (ALS)

### What is ALS?

Alternating Least Squares (ALS) is a matrix factorization technique used to solve the optimization problem above.  
Instead of updating both matrices \( P \) and \( Q \) simultaneously, ALS **alternates** between optimizing \( P \) and \( Q \) while keeping the other fixed:

1. **Fix \( Q \), solve for \( P \)** — This becomes a least-squares problem.  
2. **Fix \( P \), solve for \( Q \)** — Again, a least-squares problem.  

This process repeats until convergence.

---

### Why ALS?  

- **Quadratic Sub-problems**: When one matrix is fixed, the optimization becomes a convex quadratic problem that can be solved efficiently using least-squares methods.  
- **Parallelization**: Each user’s factors \( p_u \) and each item’s factors \( q_i \) can be updated independently, making it ideal for parallel computation (e.g., using CUDA for GPUs).  
- **Implicit Feedback**: ALS can be adapted for implicit feedback data (e.g., clicks, views) by weighting confidence for each observation.  

---

### Algorithm Steps:

1. **Initialize** \( P \) and \( Q \) with random values.  
2. **Repeat for a fixed number of iterations or until convergence**:  
    - Fix \( Q \), update all rows of \( P \) by solving the least-squares problem.  
    - Fix \( P \), update all rows of \( Q \).  
3. **Compute the error** to check convergence.  

---

### ALS Loss Function:  
The loss function in ALS is similar to the basic matrix factorization, but it splits the optimization into two parts at each iteration:

1. **Update \( P \)**: Solve \( \arg \min_P \mathcal{L}(P, Q) \)  
2. **Update \( Q \)**: Solve \( \arg \min_Q \mathcal{L}(P, Q) \)  

In each step, the least-squares solution is computed for each row independently.

---

