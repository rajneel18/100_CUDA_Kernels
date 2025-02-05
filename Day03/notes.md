# **CUDA Error Handling: Observing Failures Intentionally**

CUDA provides error-handling mechanisms that allow developers to detect and respond to failures. By intentionally triggering errors, such as excessive memory allocation, we can observe how CUDA reports and handles these failures.

---

## **1. Intentional Memory Allocation Failure**

### **Scenario:**
- Requesting an unreasonably large memory allocation (e.g., 373 GB) exceeds GPU limits.
- `cudaMalloc()` fails and returns `cudaErrorMemoryAllocation`.
- `cudaGetErrorString()` provides a human-readable error message.
- If not handled, the program may crash or behave unpredictably.



## **2. Observing CUDA Error Propagation**

### **Key Observations:**
- Once a CUDA function fails, subsequent CUDA operations may also fail.
- CUDA tracks the last error internally, and all subsequent errors may return the same failure.
- Calling `cudaGetLastError()` after a failure reveals the most recent error.

## **3. Resetting CUDA State After Failure**

### **Why Resetting is Needed?**
- If a CUDA function fails, the device may remain in an error state.
- Resetting the device clears errors and allows new CUDA operations to proceed.

### **Solution:**
- Calling `cudaDeviceReset()` resets the GPU and clears all previous errors.
- This is essential in long-running applications to avoid cumulative failures.

---

## **4. Using `cuda-memcheck` to Debug Errors**

### **What is `cuda-memcheck`?**
- A built-in CUDA tool that helps detect and debug memory errors.

### **Capabilities:**
- Detects out-of-memory issues.
- Identifies illegal memory accesses.
- Reports uninitialized memory use.
