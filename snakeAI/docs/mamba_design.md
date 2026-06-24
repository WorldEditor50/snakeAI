# Mamba (Selective State Space Model) Design

## Overview

This document describes the implementation of a simplified **Mamba** model (S6 — Selective Scan State Space Model), as introduced in the paper *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* (Gu & Dao, 2023). The MambaLayer is a drop-in replacement for RNNs, LSTMs, and Transformers as a sequence-to-sequence feature extractor in the RL library.

## Architecture

The MambaLayer implements a **selective state space model** where the transition parameters depend on the input. This is the core innovation of Mamba over traditional SSMs (like the S4 model):

### Core Equations

Given an input vector **x** ∈ ℝ^d and previous hidden state **h**_(t-1) ∈ ℝ^n:

1. **Input projection** (if d ≠ n):
   ```
   x_proj = W_in · x + b_in
   ```

2. **Selective parameters** (input-dependent):
   ```
   B_select = σ(W_B · x + b_B)         // selective input gate (sigmoid)
   Δ = softplus(W_Δ · x + b_Δ)        // selective step size
   ```

3. **Discretization** (ZOH with diagonal A):
   ```
   Ā = exp(-Δ ⊙ (1 - A_diag))          // element-wise exponential decay
   B̄ = Δ ⊙ B_select                    // discretized input
   ```

4. **State update**:
   ```
   h(t) = Ā ⊙ h(t-1) + B̄ ⊙ x_proj
   ```

5. **Output**:
   ```
   y = tanh(C · h(t) + b)
   ```

### Key Differences from Original Mamba

| Feature | Original Mamba | This Implementation |
|---------|---------------|-------------------|
| A matrix | HiPPO initialization + low-rank | Diagonal, initialized 0.9~0.99 |
| Discretization | ZOH with structured A | Element-wise ZOH with diagonal A |
| Selection mechanism | Δ, B, C all input-dependent | Δ and B only (simplified) |
| Scan | Parallel associative scan | Sequential (for BPTT) |
| Output | Activation + residual + normalization | tanh projection |

## Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `W_in` | n × d | Input projection (if d ≠ n) |
| `b_in` | n × 1 | Input bias |
| `A_diag` | n × 1 | Diagonal state transition (clamped to [0, 1]) |
| `W_B` | n × d | Selective B weight |
| `b_B` | n × 1 | Selective B bias |
| `W_Δ` | n × d | Selective step size weight |
| `b_Δ` | n × 1 | Selective step size bias |
| `C` | m × n | Output projection weight |
| `b` | m × 1 | Output bias |

Where:
- d = input dimension
- n = hidden dimension
- m = output dimension

## Training (BPTT)

The layer supports backpropagation-through-time (BPTT) by caching hidden states during the forward pass and computing gradients in reverse order. The cache stores:

- `states`: vector of `State` structs containing `h`, `y`, `B`, `Δ`, `Ā`, `B̄` at each timestep
- `cacheX`: vector of inputs **x**(t)
- `cacheE`: vector of error signals **e**(t)

### Gradient Computation

The backward pass computes gradients for all parameters:

1. **∂L/∂y**: through tanh: `δy = e(t) ⊙ tanh'(y(t))`
2. **∂L/∂C, ∂L/∂b**: `δy · h(t)^T`, `δy`
3. **∂L/∂h**: from output `C^T · δy` + propagated through Ā
4. **∂L/∂A_diag**: `δy_from_h ⊙ h(t-1) ⊙ Ā ⊙ Δ`
5. **∂L/∂Δ**: through Ā and B̄
6. **∂L/∂B_select**: through B̄ and sigmoid
7. **∂L/∂W_B, ∂L/∂W_Δ, ∂L/∂W_in, ∂L/∂b_***: standard weight gradients

## Optimizers

Supports all optimizers in the library:
- SGD
- RMSProp
- Adam

## Integration

### Type Registration

The `MambaLayer` registers as `iLayer::LAYER_MAMBA` for compatibility with `Net::backward()` in `net.hpp`.

### iLayer Interface

| Method | Implementation |
|--------|---------------|
| `forward(x, inference)` | Single timestep; caches state in training mode |
| `cacheError(e)` | Stores error for later BPTT |
| `SGD`, `RMSProp`, `Adam` | Triggers BPTT then updates parameters |
| `copyTo` | Copies all parameters |
| `softUpdateTo` | Polyak averaging for target networks |
| `write` / `read` | Serialization to file |

## Usage Example

```cpp
#include "rl/mamba.h"

using namespace RL;

// Create Mamba: input=4, hidden=16, output=3
MambaLayer mamba(4, 16, 3, true);

// Single timestep
Tensor x(4, 1);
x[0] = 0.5; x[1] = -0.3; x[2] = 0.7; x[3] = 0.1;
Tensor &out = mamba.forward(x);

// BPTT training with mini-batch
mamba.reset();
for (int t = 0; t < seqLen; t++) {
    mamba.forward(inputs[t]);
    mamba.cacheError(Loss::MSE::df(mamba.o, targets[t]));
}
mamba.RMSProp(0.01f, 0.9f, 0.0f, true);

// Inference
mamba.forward(x, true);   // no caching
```

## Test Suite

The test file `test/test_mamba.cpp` runs 10 tests:

1. **Constructor & Type** — verifies dimensions and type registration
2. **Single Forward Pass** — validates tanh output range [-1, 1]
3. **Sequential State Persistence** — verifies hidden state evolves
4. **Selective B/Δ Behavior** — different inputs produce different gates
5. **BPTT Gradient Flow** — non-zero gradients for all parameters
6. **MSE Training** — trains on z = sin(x²+y²) and validates loss reduction
7. **Reset** — clears cache and state
8. **copyTo / softUpdateTo** — parameter copy and Polyak averaging
9. **Save/Load** — serialization round-trip
10. **Inference Mode** — no caching when `inference=true`
