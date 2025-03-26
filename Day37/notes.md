# CUDA Graphs (Markdown Notes)

## Concept

CUDA Graphs allow you to define and launch a sequence of CUDA operations (kernels, memory copies) as a single unit, reducing CPU overhead and improving performance.

## Key Concepts

* **Graph Creation:**
    * Define a sequence of CUDA operations as nodes in a graph.
    * Nodes represent kernels, memory copies, or other CUDA operations.
    * Edges define dependencies between nodes.
* **Graph Instantiation:**
    * Create an executable instance of the graph.
* **Graph Launch:**
    * Launch the graph instance, executing all operations in the defined sequence.
    * Reduces CPU overhead compared to launching individual operations.
* **Graph Updates:**
    * Modify graph parameters (e.g., kernel arguments) without rebuilding the entire graph.
    * Allows for efficient iteration and dynamic behavior.

## CUDA Graph APIs

* `cudaGraphCreate()`: Creates a CUDA graph.
* `cudaGraphAddKernelNode()`: Adds a kernel execution node.
* `cudaGraphAddMemcpyNode()`: Adds a memory copy node.
* `cudaGraphInstantiate()`: Creates an executable instance of the graph.
* `cudaGraphLaunch()`: Launches the graph instance.
* `cudaGraphExecKernelNodeSetParams()`: Changes the kernel parameters of an executed graph.

