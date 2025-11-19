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

Grid search experiments on Taxi-v3 (264 configs, after removing Savgol and fixing PCHIP) revealed:
- **Average MAE**: Polar (2.90) ≈ Linear (2.93) ≈ PCHIP (2.98) << Lanczos (5.34)
- **Average MSE**: Polar (37.2) << Lanczos (62.9) < Linear (73.5) << PCHIP (182.6)
- **Robustness (MSE std)**: Polar (71.7) << Linear (338.6) << PCHIP (1236.3!)
- **Surprise**: MAE alone is misleading - PCHIP looks similar to polar/linear but has **extreme instability**
- **In excellent configs** (<0.1 MAE): Linear (25) ≈ Polar (25) ≈ PCHIP (22) >> Lanczos (3)
- **All top 5 configs** use polar interpolation
- Best config achieves MAE ≈ 10⁻¹⁵ with polar + adaptive/piecewise-centered + Gaussian

**Critical insight**: **PCHIP has hidden instability** - despite good average MAE (2.98), it has massive MSE variance (std = 1236.3). Some PCHIP configs work well, but others produce catastrophic errors. Polar has both good average performance AND low variance (MSE std = 71.7) - it's consistently excellent.

**Key insight**: For CVI characteristic functions, simple interpolation methods (linear/polar) work just as well as sophisticated ones (PCHIP) on average, but are far more **stable and predictable**. The CF structure is smooth enough that linear/polar suffices. Polar achieves the absolute best peak performance AND lowest variance when paired with adaptive grids and Gaussian collapse.

**Recommendation**: Use **polar** as default (best peak performance, lowest variance, appears in all top configs). Linear is a solid alternative. **Avoid PCHIP** despite similar MAE - hidden instability makes it unreliable. Avoid Lanczos - consistently worse.

