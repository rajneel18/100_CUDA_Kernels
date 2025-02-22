import numpy as np
import torch
import tensorflow as tf
import timeit

# Generate a large input array
N = 10000
x = np.random.rand(N).astype(np.float32)

# NumPy Softmax
def softmax_numpy(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

# PyTorch Softmax
x_torch = torch.tensor(x)
def softmax_torch():
    torch.nn.functional.softmax(x_torch, dim=0)

# TensorFlow Softmax
x_tf = tf.constant(x)
def softmax_tf():
    tf.nn.softmax(x_tf)

# Measure execution time
num_runs = 100
time_numpy = timeit.timeit(lambda: softmax_numpy(x), number=num_runs) / num_runs
time_torch = timeit.timeit(softmax_torch, number=num_runs) / num_runs
time_tf = timeit.timeit(softmax_tf, number=num_runs) / num_runs

# Print results in seconds and milliseconds
print(f"NumPy Softmax Time: {time_numpy:.6f} sec ({time_numpy * 1000:.3f} ms)")
print(f"PyTorch Softmax Time: {time_torch:.6f} sec ({time_torch * 1000:.3f} ms)")
print(f"TensorFlow Softmax Time: {time_tf:.6f} sec ({time_tf * 1000:.3f} ms)")
