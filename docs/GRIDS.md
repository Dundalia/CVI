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

Grid search experiments on Taxi-v3 (264 configs, after removing Savgol and fixing PCHIP) revealed:
- **Adaptive**: MAE ~1.61, MSE ~13.1 (BEST - zero hyperparameters!)
- **Piecewise-centered**: MAE ~1.76, MSE ~18.6 (excellent, but requires tuning)
- **Logarithmic** (λ=2.0): MAE ~1.97, MSE ~16.7 (good, single parameter)
- **Chebyshev**: MAE ~4.98, MSE ~51.0 (poor - dense at wrong locations)
- **Uniform**: MAE ~6.09, MSE ~260.2 (UNSTABLE - see below)

**Key findings**:
- Adaptive achieves best performance (20/75 excellent configs) with **zero hyperparameters** - automatically concentrates density near ω=0
- All top 5 configs use adaptive or piecewise-centered grids with polar + Gaussian
- Best configs achieve MAE ≈ 10⁻¹⁵ (exact match with VI, numerical precision limit)
- **Larger K consistently better AND more stable**: K=512 (MAE ~2.5, MSE std 55) > K=256 (MAE ~3.3, MSE std 61) > K=128 (MAE ~4.8, MSE std 1104!)
- **Uniform grid is dangerously unstable**: MSE std = 1216 (4.7x its mean!) - contains "landmine" configurations with catastrophic errors
- **Adaptive has lowest variance**: MSE std = 20.6 (only 1.6x mean) - consistently good across all settings

**Recommendation**: Use **adaptive** as default (best MAE, lowest variance, zero tuning). Use **logarithmic** if you want explicit control via λ. **STRONGLY avoid uniform** - it's not just worse on average but dangerously unstable with hidden failure modes. Use K=512 for best accuracy and stability, minimum K=256.

