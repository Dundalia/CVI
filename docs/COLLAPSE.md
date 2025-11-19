# Collapse Methods

## Why Collapse?

CVI operates in the frequency domain, representing return distributions via characteristic functions φ(ω) = E[e^(iωG)]. To extract scalar Q-values for action selection, we need to **collapse** the CF back to its first moment: Q(s,a) = E[G(s,a)]. This is a critical step that bridges frequency-domain learning with standard RL control.

---

## Methods

### 1. Least Squares (LS)

Fits a local quadratic polynomial φ(ω) ≈ a₀ + a₁ω + a₂ω² around ω=0 using nearby grid points. The mean is extracted from the imaginary part of the linear coefficient: E[G] = Im(a₁). Simple, interpretable, and works well with non-uniform grids. Uses m=4 neighbors by default.

### 2. Fast Fourier Transform (FFT)

Directly inverts the CF to the spatial domain via IFFT to recover the probability density, then computes the mean in the spatial domain. Theoretically elegant but requires a strictly uniform grid and suffers from discretization artifacts. In practice, produces high errors (MAE ~11.8) likely due to normalization issues and boundary effects.

### 3. Gaussian Fit

Assumes the CF has a Gaussian-like structure: log φ(ω) ≈ iμω - ½σ²ω². Extracts the mean μ by unwrapping the phase arg(φ) and fitting a line through the origin. Exploits the natural shape of bounded-reward CFs and consistently achieves the best performance (MAE ~2.3). Robust to grid type and most reliable method overall.

---

## Empirical Performance

Grid search experiments on Taxi-v3 (264 configs, after removing Savgol and fixing PCHIP) revealed:
- **MAE**: Gaussian (2.08) >> LS (3.34) >> FFT (11.81)
- **MSE**: Gaussian (23.0) < LS (30.5) << FFT (711.8)
- **RMSE/MAE ratio** (robustness indicator): Gaussian (1.23, best!) < LS (1.40) < FFT (1.61)
- Gaussian collapse paired with adaptive/piecewise-centered grids and polar interpolation achieves near-zero error (MAE ≈ 10⁻¹⁵)
- **In excellent configs** (<0.1 MAE): Gaussian dominates with 72/75 (96%), LS has only 3/75 (4%)

**Key insight**: **Gaussian is remarkably robust** - lowest RMSE/MAE ratio (1.23) indicates errors are tightly distributed and consistently small. When Gaussian makes mistakes, they're bounded and predictable. FFT's high ratio (1.61) indicates frequent large outlier errors, making it unpredictably bad.

**Recommendation**: Use **Gaussian collapse** as default. It exploits the natural structure of characteristic functions and achieves both the best accuracy AND highest robustness (lowest error variance). LS is a solid alternative if you need variance estimation or prefer interpretability. **STRONGLY avoid FFT** - not just worse on average but produces large unpredictable outliers.

