# Grid Construction Methods for CVI

## Why Grid Construction Matters

CVI operates in the frequency domain by discretizing the continuous ω-space onto a finite grid of K points in [-W, W]. The choice of grid directly impacts: (1) accuracy of moment extraction near ω=0, (2) interpolation quality for scaled frequencies γω, and (3) computational efficiency. An optimal grid concentrates points where the CF contains the most information while minimizing total grid size K.

**What we construct**: A set of K frequency values {ω₁, ..., ωₖ} in [-W, W] on which we evaluate and store the characteristic function φ(s, ω). The distribution of these points fundamentally affects CVI's ability to accurately represent return distributions.

---

## Implemented Methods

### 1. Uniform
Evenly spaced points with constant spacing Δω = 2W/(K-1).

**Approach**: `ωₖ = -W + k·Δω` for k = 0, ..., K-1.

**Pros**: Simple; works well with FFT-based methods; predictable interpolation behavior; no hyperparameters.

**Cons**: Wastes points in tails where CF may be less informative; equal spacing doesn't match the importance of ω=0 for moment extraction.

### 2. Piecewise-Centered
Dense uniform grid in [-w_center, w_center], coarser uniform grid in tails.

**Approach**: Allocate `frac_center` fraction of points uniformly in center region, remaining points in two tail regions.

**Pros**: Concentrates points near ω=0 where mean extraction happens; significantly better than uniform (3x lower error in experiments).

**Cons**: Requires manual tuning of w_center and frac_center hyperparameters; sharp density transitions at boundaries may affect interpolation.

### 3. Logarithmic
Exponential spacing: dense near 0, logarithmically sparse in tails.

**Approach**: `ωₖ = ±W·(exp(k·λ/K_half) - 1)/(exp(λ) - 1)` with decay parameter λ.

**Pros**: Single tunable parameter λ controls density; smooth density gradient (no sharp transitions); mathematically principled for decaying CFs.

**Cons**: Performance depends on CF decay rate; may allocate too few points in mid-range frequencies.

### 4. Chebyshev
Uses zeros of Chebyshev polynomials: `ωₖ = W·cos(π(2k-1)/(2K))`.

**Approach**: Standard Chebyshev nodes of the first kind, scaled to [-W, W].

**Pros**: Minimizes Runge phenomenon (polynomial interpolation oscillations); theoretically optimal for polynomial approximation; zero hyperparameters.

**Cons**: Denser near boundaries ±W (not ω=0), which may not align with CVI needs; less intuitive point distribution.

### 5. Adaptive
Three-region grid: very dense near 0 (50% points in 20% range), medium density in middle, sparse in tails.

**Approach**: Pre-allocated density regions with fixed boundaries at 0.1W and 0.4W.

**Pros**: Balances accuracy near ω=0 with coverage of full range; no manual tuning of boundaries; generally robust across problems.

**Cons**: Fixed region boundaries may not be optimal for all MDPs; not truly adaptive (doesn't use feedback from CF curvature during iteration).

---

## Empirical Performance

Grid search experiments on Taxi-v3 (342 configs with all 5 strategies) revealed:
- **Piecewise-centered** (MAE ~3.0) ≈ **logarithmic** (MAE ~3.0) ≈ **adaptive** (MAE ~3.1) >> **uniform** (MAE ~5.5) >> **chebyshev** (MAE ~38.6, worst!)
- Chebyshev dramatically underperforms despite theoretical advantages for interpolation (boundary density is wrong for CVI)
- Logarithmic and adaptive successfully match piecewise-centered without manual hyperparameter tuning
- Best overall config: piecewise-centered + polar + Gaussian (MAE ≈ 10⁻¹⁵), but all three top strategies can achieve near-zero error with right collapse/interpolation
- Grid size K shows complex interaction effects: best configs achieve near-zero error regardless of K, but poor grid strategies (especially Chebyshev) degrade significantly with larger K

**Recommendation**: Use **logarithmic** (single parameter λ=2.0) or **adaptive** (zero parameters) for simplicity, or **piecewise-centered** if you want the proven best. Avoid Chebyshev for CVI. K=256 provides good balance of accuracy and efficiency.

