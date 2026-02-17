# Recommendation: High-Performance Probabilistic Engine

## 1. Executive Summary
For the "System 2" Empirical Verifier, we recommend **NumPyro** (backed by **JAX**) as the industry-standard, high-performance library for probabilistic simulations and optimization.

## 2. Rationale
The choice is driven by the need for speed (high-iteration Monte Carlo), automatic differentiation (for gradient-based optimization), and functional purity (which aligns with formal reasoning paradigms).

| Feature | NumPy/SciPy | PyTorch | JAX / NumPyro |
| :--- | :--- | :--- | :--- |
| **Performance** | Low (CPU only) | High (GPU) | **Very High** (XLA Compilation) |
| **Paradigm** | Imperative | Object-Oriented | **Functional** (Math-like) |
| **Probabilistic** | Manual | Pyro (Mature) | **NumPyro** (Fastest MCMC) |
| **Verification** | Hard to inspect | Stateful objects | **Pure functions** (easier to map to Lean) |

## 3. Detailed Comparison

### A. PyTorch (and Pyro)
*   **Pros:** Dominant industry adoption, easy debugging, dynamic graph.
*   **Cons:** Heavier runtime overhead, object-oriented state management can obscure formal logic flows.

### B. JAX (and NumPyro)
*   **Pros:**
    *   **XLA Compilation:** Compiles Python/NumPy code into highly optimized machine code (CPU/GPU/TPU).
    *   **Auto-Vectorization (`vmap`):** Trivial parallelization of simulations, essential for the "10,000 iterations" requirement.
    *   **Statelessness:** JAX requires pure functions (no side effects), which mirrors the mathematical functions defined in Lean 4.
*   **Cons:** Steeper learning curve (immutable arrays), but this constraint actually enforces better code quality for a formal system.

## 4. Implementation Strategy

We will standardize on **JAX** for the raw computation and **NumPyro** for the probabilistic modeling primitives.

### Example Code Structure (for `src/verifiers/python_verifier.py`)

```python
import jax.numpy as jnp
from jax import random, vmap

def simulation_kernel(key, parameters):
    # Pure function: State -> Outcome
    ...

def run_simulation(num_samples=10000):
    key = random.PRNGKey(0)
    keys = random.split(key, num_samples)
    results = vmap(simulation_kernel, in_axes=(0, None))(keys, params)
    return jnp.mean(results)
```

## 5. Conclusion
**JAX/NumPyro** is the optimal choice. It offers the raw speed of C++ via XLA, the usability of Python, and a functional design philosophy that aligns perfectly with the project's formal verification goals.
