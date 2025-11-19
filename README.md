# CVI-RL: Characteristic Value Iteration for Reinforcement Learning

Frequency-domain reinforcement learning using characteristic functions on tabular MDPs.

## Overview

This repository implements **Characteristic Value Iteration (CVI)**, a novel approach to RL that operates in the frequency domain. Instead of computing scalar value functions, CVI represents return distributions via their characteristic functions φ(ω) = E[e^(iωG)], enabling distributional policy evaluation through CF Bellman operators.

## Installation

```bash
# Clone the repository
cd cvi_rl

# Install dependencies
pip install -r requirements.txt
```

## Experiment Suite (`experiment_suite.ipynb`)

The notebook contains a comprehensive hyperparameter grid search evaluating 264 successful CVI configurations on Taxi-v3:

- **Variables tested**: 
  - Grid strategies: uniform, piecewise-centered, logarithmic, chebyshev, adaptive
  - Frequency ranges (W): 10.0, 20.0
  - Grid sizes (K): 128, 256, 512
  - Interpolation methods: linear, polar, pchip, lanczos
  - Collapse methods: ls, fft, gaussian

- **Key Results**: 
  - **Best config**: adaptive + polar + gaussian (MAE ≈ 10⁻¹⁵, exact match with VI)
  - **Grid strategies**: adaptive (1.61, zero hyperparameters!) > piecewise (1.76) > logarithmic (1.97) > chebyshev (4.98) > uniform (6.09)
  - **Collapse methods**: gaussian (2.08, 96% of excellent configs) >> ls (3.34) >> fft (11.81)
  - **Interpolation**: polar (2.90) ≈ linear (2.93) ≈ pchip (2.98) << lanczos (5.34)
  - **Grid size**: Larger K better: K=512 (2.50) > K=256 (3.30) > K=128 (4.81)

### Grid Strategy Comparison

The choice of ω-grid significantly impacts performance. Below shows the distribution of grid points for each strategy:

![Grid Comparison](grid_comparison.png)

**Key insight**: The **adaptive grid strategy** achieves the best performance (MAE 1.61) with **zero hyperparameters** by automatically concentrating density near ω=0 where moment extraction occurs. Combined with polar interpolation and Gaussian collapse, CVI can exactly match classical Value Iteration (MAE ≈ 10⁻¹⁵).

## Key Methods Implemented

### Interpolation (for V(s, γω))
- **Polar**: Magnitude/phase interpolation (best peak performance, appears in all top configs)
- **Linear**: Cartesian interpolation of real/imaginary parts (equally good average performance)
- **PCHIP**: Monotonicity-preserving cubic (equally good average performance)
- **Lanczos**: Windowed sinc interpolation (underperforms despite theoretical advantages)

### Collapse (extracting mean from CF)
- **Gaussian**: Phase unwrapping with linear regression (best performer, 96% of excellent configs)
- **LS**: Least-squares quadratic fit around ω=0 (solid alternative)
- **FFT**: Inverse Fourier transform to spatial domain (unreliable)

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **`STATE.md`**: Overall implementation status, design choices, experimental findings, and known limitations
- **`GRIDS.md`**: Grid construction methods (uniform, piecewise-centered, logarithmic, chebyshev, adaptive)
- **`INTERPOLATION.md`**: Interpolation methods for evaluating V(s, γω) at off-grid frequencies
- **`COLLAPSE.md`**: Collapse methods for extracting scalar Q-values from characteristic functions

