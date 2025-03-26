#  Block-Wise Quasi-Newton Updates in CUDA

##  Introduction  
**Block-Wise Quasi-Newton Updates** optimize the **L-BFGS (Limited-memory BFGS) algorithm** by dividing the problem into **blocks** and updating local curvature estimates using **shared memory**. This enhances **data locality** and **reduces global memory latency**.

---

##  Key Concepts  

### **L-BFGS Basics**
- **L-BFGS (Limited-memory BFGS)** is a quasi-Newton method used for optimization.
- It **approximates the Hessian** using only a few past updates of gradients.
- Works well for **high-dimensional optimization problems**.

###  **Why Use Block-Wise Updates?**
- **Standard L-BFGS** requires **global memory reads/writes**, which can be slow.
- **Block-wise updates** use **shared memory** to reduce latency.
- Allows **hierarchical reduction** to combine block-level updates efficiently.

###  **CUDA Implementation Strategy**
- Divide computation into **CUDA blocks**.
- Each **block** loads `s` and `y` vectors into **shared memory**.
- Compute **local curvature updates** using **parallel two-loop recursion**.
- Use **hierarchical reduction** to combine results efficiently.

