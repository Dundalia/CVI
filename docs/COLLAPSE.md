# Collapse Methods for CVI

## Why Collapsing is Needed

CVI computes the full characteristic function φ(ω) = E[e^(iωG)] representing the return distribution. To derive scalar Q-values for greedy action selection, we must **collapse** the CF to extract the mean return: E[G] = Im(φ'(0)). This is the "collapse" step that bridges distributional evaluation to classical control.

**What we collapse**: Complex-valued characteristic functions φ(ω) defined on a discrete grid. The goal is to accurately estimate the derivative φ'(0) from sampled values, then extract its imaginary part as the mean.

---

## Implemented Methods

### 1. LS (Least Squares)
Fits a local quadratic φ(ω) ≈ a₀ + a₁ω + a²ω² around ω=0 using a window of ±m grid points.

**Approach**: Use least-squares regression on complex coefficients, extract a₁, return Im(a₁).

**Pros**: Robust to noise; can also estimate variance from a₂; simple and interpretable; consistent performance across problems.

**Cons**: Requires manual tuning of window size m; assumes local smoothness; quadratic approximation may not capture complex CF shapes.

### 2. FFT (Fast Fourier Transform)
Applies inverse FFT to transform φ(ω) from frequency domain to spatial domain, obtaining the PDF/PMF of returns, then computes mean directly.

**Approach**: `pdf = IFFT(φ(ω))`, then `E[G] = Σ x·pdf(x)`.

**Pros**: Theoretically elegant; recovers full distribution (not just mean); no local approximation needed.

**Cons**: Requires uniform grid; highly sensitive to aliasing and normalization; poor empirical performance (worst in experiments with MAE ~11.8 vs ~1.5 for Gaussian).

### 3. Gaussian (Phase Unwrapping)
Assumes locally Gaussian CF: log φ(ω) ≈ iμω - ½σ²ω², then fits a line to unwrapped phase to extract mean μ.

**Approach**: Compute phase = unwrap(arg(φ(ω))), fit linear regression phase ≈ μω, return μ.

**Pros**: Leverages the structure of characteristic functions; most accurate in experiments (best performer with MAE ~1.5); naturally handles Gaussian-like return distributions.

**Cons**: Assumes local Gaussian structure (may fail for multimodal or heavy-tailed distributions); phase unwrapping can be unstable near discontinuities.

### 4. Savitzky-Golay
Applies a Savitzky-Golay smoothing filter with derivative estimation to Im(φ(ω)), then evaluates at ω=0.

**Approach**: Use `scipy.signal.savgol_filter` with `deriv=1` on Im(φ), extract value at ω=0.

**Pros**: Smooths noise while computing derivative; configurable window and polynomial order; well-established numerical method.

**Cons**: Requires uniform or nearly-uniform grid spacing; tuning window_length and polyorder can be tricky; moderate performance (MAE ~4.3).

---

## Empirical Performance

Grid search experiments on Taxi-v3 (342 configs) revealed:
- **Gaussian** (MAE ~2.3) >> **LS** (MAE ~3.6) >> **FFT** (MAE ~11.8) >> **Savgol** (MAE ~28.6)
- Gaussian collapse paired with piecewise-centered grids and polar interpolation achieves near-zero error (MAE ≈ 10⁻¹⁵)
- FFT is unreliable despite theoretical appeal; likely due to discretization artifacts and normalization issues
- Savgol performs much worse than expected (high variance), possibly due to sensitivity to non-uniform grids from new strategies

**Recommendation**: Use **Gaussian collapse** as default. It exploits the natural structure of characteristic functions and consistently achieves the best accuracy. LS is a solid alternative if you need variance estimation or prefer interpretability. Avoid Savgol and FFT.

