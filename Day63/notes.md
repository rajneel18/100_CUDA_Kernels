# Spiking Neural Networks in CUDA 

## Overview
Spiking Neural Networks (SNNs) are a class of artificial neural networks that more closely mimic biological neurons. Unlike traditional ANNs, which use continuous activations, SNNs use discrete spikes to process information.

In this implementation, we use CUDA to accelerate the simulation of an SNN. We will focus on:
- **Leaky Integrate-and-Fire (LIF) neurons**
- **Synaptic weight updates**
- **Parallel spike propagation using CUDA**

## Implementation Steps
1. **Initialize Neurons:** Define membrane potentials, thresholds, and refractory periods.
2. **Update Neuron States:** Compute new potentials and check for spiking neurons.
3. **Propagate Spikes:** Use CUDA to efficiently distribute spike signals.
4. **Update Synaptic Weights:** Apply learning rules (STDP - Spike-Timing Dependent Plasticity).
