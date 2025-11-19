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

The notebook contains a comprehensive hyperparameter grid search evaluating 128 different CVI configurations on Taxi-v3:

- **Variables tested**: Grid strategies (uniform, piecewise-centered), frequency ranges, grid sizes (128-512), interpolation methods (linear, polar, pchip, lanczos), and collapse methods (ls, fft, gaussian, savgol)
- **Key Results**: Best configuration (piecewise-centered + polar + gaussian) achieves near-zero error (MAE ≈ 10⁻¹⁵), exactly recovering VI Q-values

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

See `docs/STATE.md` for detailed information on implementation design choices, mathematical formulations, experimental findings, and known limitations.

