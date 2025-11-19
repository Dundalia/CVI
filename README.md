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

The notebook contains a comprehensive hyperparameter grid search evaluating 342 different CVI configurations on Taxi-v3:

- **Variables tested**: 
  - Grid strategies: uniform, piecewise-centered, logarithmic, chebyshev, adaptive
  - Frequency ranges (W): 10.0, 20.0
  - Grid sizes (K): 128, 256, 512
  - Interpolation methods: linear, polar, pchip, lanczos
  - Collapse methods: ls, fft, gaussian, savgol

- **Key Results**: 
  - Best config: piecewise-centered + polar + gaussian (MAE ≈ 10⁻¹⁵, exact match with VI)
  - Grid strategies: logarithmic ≈ adaptive ≈ piecewise-centered (~3.0) >> uniform (~5.5) >> chebyshev (~38.6, fails badly)
  - Collapse methods: gaussian (2.3) >> ls (3.6) >> fft (11.8) >> savgol (28.6)
  - Interpolation: lanczos most consistent (5.3), polar achieves best peak performance with right combinations

### Grid Strategy Comparison

The choice of ω-grid significantly impacts performance. Below shows the distribution of grid points for each strategy:

![Grid Comparison](grid_comparison.png)

**Key insight**: Dense sampling near ω=0 is critical for accurate moment extraction. Logarithmic and adaptive strategies successfully match piecewise-centered performance without manual hyperparameter tuning.

## Key Methods Implemented

### Interpolation (for V(s, γω))
- **Linear**: Cartesian interpolation of real/imaginary parts
- **Polar**: Magnitude/phase interpolation (best performer)
- **PCHIP**: Monotonicity-preserving cubic
- **Lanczos**: Windowed sinc interpolation

### Collapse (extracting mean from CF)
- **LS**: Least-squares quadratic fit around ω=0
- **FFT**: Inverse Fourier transform to spatial domain
- **Gaussian**: Phase unwrapping with linear regression (best performer)
- **Savitzky-Golay**: Smoothing filter with derivative estimation

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **`STATE.md`**: Overall implementation status, design choices, experimental findings, and known limitations
- **`GRIDS.md`**: Grid construction methods (uniform, piecewise-centered, logarithmic, chebyshev, adaptive)
- **`INTERPOLATION.md`**: Interpolation methods for evaluating V(s, γω) at off-grid frequencies
- **`COLLAPSE.md`**: Collapse methods for extracting scalar Q-values from characteristic functions

