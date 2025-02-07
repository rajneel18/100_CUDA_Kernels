# CUDA Parallel Prefix Sum (Scan) Notes  

## What is Prefix Sum?  
- **Prefix Sum** (*Scan**) calculates cumulative sums of an array.  
- For each element in the array, it replaces the value with the sum of all previous elements, including itself.  


## Why Use Parallel Prefix Sum?   
- **Parallel Scan:** Reduces computation time by processing multiple elements simultaneously on a GPU instead of doing the same task simultaneously.  
- **Widely used in:**  
    - Sorting  
    - Histogram computation  
    - Stream compaction  
    - Machine learning algorithms (e.g., softmax, cumulative probability)  

---

## Parallel Prefix Sum Phases  

### 1. Up-Sweep (Reduction Phase)  
- Builds a **tree structure** of partial sums.  
- Each thread adds elements at a specific **stride** and stores the result at higher positions.  
- After this phase, the **last element** contains the total sum of the array.  

### 2. Set Last Element to Zero  
- This step prepares for the second phase by **setting the last element to zero**.  

### 3. Down-Sweep Phase  
- Converts partial sums into **prefix sums**.  
- Each thread computes the correct prefix sum using the previously built partial sums.  

---

## CUDA Implementation Key Concepts  

- **Up-Sweep and Down-Sweep:** These phases are implemented using loops with different strides.  
- **Thread Synchronization:** `__syncthreads()` ensures that all threads complete their current work before moving to the next step.  
- **Memory Access:** Each thread processes specific elements based on its **index and stride**.  

---

## Applications of Prefix Sum  

- **Parallel sorting** (e.g., radix sort)  
- **Parallel filtering** (removing invalid elements)  
- **Cumulative sums** in large datasets  
- **Financial and statistical calculations**  
