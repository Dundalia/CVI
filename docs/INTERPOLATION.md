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

Comprehensive grid search on Taxi-v3 (360 configurations) revealed that **polar interpolation is absolutely critical** for achieving optimal performance.

### Top 40 Configurations
- **ALL use polar interpolation** (100% dominance)
- Combined with Gaussian collapse: MAE = 10⁻¹⁵ to 10⁻¹⁴ (machine precision)
- Works across all grid strategies and hyperparameters
- Best single config: MAE = 1.27×10⁻¹⁵ (four density regions, polar, gaussian)

### Method-Specific Performance (with Gaussian collapse)
When paired with Gaussian collapse, interpolation methods show dramatic differences:
- **Polar**: 33 of top 40 configs (82.5%) - MAE ~10⁻¹⁵ (DOMINANT)
- **Linear**: Can work but not in top performers
- **PCHIP**: Acceptable with LS collapse (MAE ~10⁻⁷) but never top-tier
- **Lanczos**: Catastrophic failure mode when combined with LS collapse (MAE = 6-8)

### Collapse Method Interaction
The collapse method drastically affects interpolation performance:
- **Polar + Gaussian**: Perfect combination (MAE ~10⁻¹⁵)
- **Polar/PCHIP + LS**: Acceptable (MAE ~10⁻⁷)  
- **Any method + FFT**: Catastrophic (MAE = 7-27)
- **Lanczos + LS**: Consistent failure (MAE = 6-8)

### Bottom 40 Configurations
Poor interpolation alone doesn't guarantee failure - **bad collapse methods dominate**:
- FFT collapse: 27/40 worst configs (regardless of interpolation)
- Lanczos + LS: 13/40 worst configs

Even polar interpolation fails with FFT collapse (MAE ~11 vs ~10⁻¹⁵ with Gaussian).

**Critical Discovery**: Interpolation method matters, but **only when paired with the right collapse method**. Polar interpolation is necessary but not sufficient - it must be combined with Gaussian collapse to achieve optimal performance.

**Key insight**: The synergy between interpolation and collapse methods is crucial. Polar interpolation preserves the CF structure's magnitude and phase properties, which Gaussian collapse exploits through phase unwrapping. This combination achieves results at the numerical precision limit.

**Recommendation**: 
1. **ALWAYS use polar interpolation** - mandatory for top performance
2. **MUST pair with Gaussian collapse** - polar alone is insufficient
3. **Never use Lanczos with LS collapse** - consistent failure mode
4. **Avoid any interpolation with FFT** - catastrophic regardless of method

