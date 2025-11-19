# Interpolation Methods for CVI

## Why Interpolation is Needed

In Characteristic Value Iteration, the CF Bellman operator requires evaluating V(s, γω) where γ is the discount factor. Since we discretize the frequency domain on a finite grid of ω values, the scaled frequency γω will generally **not** lie on our grid points. Interpolation is necessary to estimate V(s, γω) from the known values at nearby grid points.

**What we interpolate**: Complex-valued functions φ(ω) representing characteristic functions. Each interpolation occurs in the complex plane (ℂ), requiring special consideration for how we handle real/imaginary parts or magnitude/phase.

---

## Implemented Methods

### 1. Linear (Cartesian)
Applies standard linear interpolation independently to the real and imaginary parts of φ(ω).

**Approach**: `φ(ω) = Re(φ)_interpolated + i·Im(φ)_interpolated`

**Pros**: Fast, simple, works well for smooth CFs.

**Cons**: Can violate |φ(ω)| ≤ 1 between grid points; treats magnitude and phase independently which may not respect CF structure.

### 2. Polar
Interpolates magnitude |φ(ω)| and unwrapped phase arg(φ(ω)) separately, then reconstructs the complex number.

**Approach**: `φ(ω) = |φ|_interpolated · exp(i·phase_interpolated)`

**Pros**: Better preserves the constraint |φ(ω)| ≤ 1; respects the polar structure of complex exponentials. Empirically best performer in experiments.

**Cons**: Phase unwrapping can be unstable if φ(ω) has discontinuities; slightly more computational overhead.

### 3. PCHIP (Piecewise Cubic Hermite)
Monotonicity-preserving piecewise cubic interpolation applied to real and imaginary parts independently.

**Approach**: Uses scipy's `PchipInterpolator` on Re(φ) and Im(φ) separately.

**Pros**: Avoids overshoots and oscillations (Runge phenomenon) common with cubic splines; smoother than linear.

**Cons**: Does not preserve CF validity constraints; more complex than linear but empirically not better than polar.

### 4. Lanczos
Windowed sinc interpolation using the Lanczos kernel: L(x) = sinc(x)·sinc(x/a) for |x| < a.

**Approach**: For each target point, computes weighted sum over ±a neighboring grid points using the Lanczos kernel.

**Pros**: Theoretically optimal for bandlimited signals; best spectral properties for uniform grids.

**Cons**: Computationally expensive (requires window of 2a+1 points per evaluation); designed for uniform grids; empirically not better than simpler methods for CVI.

---

## Empirical Performance

Grid search experiments on Taxi-v3 (342 configs) revealed:
- **Average MAE**: Lanczos (5.3) < Linear (9.9) < Polar (14.4) < PCHIP (17.6)
- **Best config**: Uses **polar** interpolation despite higher average MAE (interaction effect with grid/collapse)
- **Explanation**: Polar achieves near-zero error with optimal combinations (piecewise-centered + Gaussian) but performs poorly with mismatched settings. Lanczos is more consistently good across all configs but never reaches the absolute best performance.
- High standard deviations for polar/PCHIP/linear indicate strong parameter interactions

**Recommendation**: Use **lanczos** for robustness across settings, or **polar** if paired with piecewise-centered/logarithmic grids and Gaussian collapse for best possible accuracy.

