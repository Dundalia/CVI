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

Comprehensive grid search on Taxi-v3 (360 configurations) revealed that **collapse method is THE most critical factor** - even more important than interpolation or grid strategy.

### Top 40 Configurations
- **33 use Gaussian collapse** (82.5%) - MAE ~10⁻¹⁵ (machine precision)
- **7 use LS collapse** (17.5%) - MAE ~10⁻⁷ (acceptable but 8 orders of magnitude worse)
- **0 use FFT collapse** (catastrophic failure)

When paired with polar interpolation:
- **Gaussian**: MAE = 1.27×10⁻¹⁵ to 6.08×10⁻¹⁴ (DOMINANT - achieves numerical precision)
- **LS**: MAE = 1.07×10⁻⁷ to 9.13×10⁻⁶ (acceptable as fallback)
- **FFT**: MAE = 7 to 27 (CATASTROPHIC - avoid at all costs)

### Bottom 40 Configurations
**FFT collapse dominates failures** with devastating impact:
- **27 of worst 40 configs** (67.5%) use FFT
- MAE ranges from 7 to 27 (vs. 10⁻¹⁵ for Gaussian)
- Fails across **ALL grid strategies and interpolation methods**
- Particularly catastrophic with uniform grid (MAE = 26.6, worst config)

Even with optimal polar interpolation and advanced grids, FFT produces terrible results.

### LS Collapse Behavior
LS shows a specific failure mode:
- Works acceptably with polar/PCHIP interpolation (MAE ~10⁻⁷)
- **Catastrophically fails with Lanczos** interpolation (MAE = 6-8)
- 13 of worst 40 configs are Lanczos + LS combinations

### Method Synergy
Collapse method effectiveness depends on interpolation pairing:
- **Gaussian + Polar**: Perfect synergy - Gaussian's phase unwrapping exploits polar's magnitude/phase structure
- **LS + Polar/PCHIP**: Acceptable - local polynomial fit works with smooth interpolation
- **LS + Lanczos**: Total failure - Lanczos oscillations break LS fitting
- **FFT + Anything**: Catastrophic - discretization artifacts and normalization issues

**Critical Discovery**: Gaussian collapse achieves 8+ orders of magnitude better accuracy than alternatives. The performance gap is not gradual - it's a cliff. FFT is not "somewhat worse" - it's completely unusable.

**Key insight**: Gaussian collapse works by unwrapping the phase of the CF and fitting a line through the origin, naturally exploiting the CF structure for Gaussian-like distributions. This fundamental alignment with CF properties explains its dramatic superiority.

**Recommendation**: 
1. **ALWAYS use Gaussian collapse** - non-negotiable for optimal performance
2. **LS is acceptable as fallback** - if Gaussian unavailable, but expect ~8 orders of magnitude worse accuracy
3. **NEVER use FFT collapse** - catastrophic across all configurations
4. **Avoid LS with Lanczos** - specific failure mode to watch for

