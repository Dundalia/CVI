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

### 2. Two Density Regions
Dense uniform grid in center region, coarser uniform grid in tails.

**Approach**: Allocate `frac_center` (default: 90%) of points uniformly in center region `[-w_center·W, w_center·W]` (default: w_center=0.3), remaining points in two tail regions.

**Pros**: Concentrates points near ω=0 where mean extraction happens; simple two-tier density structure; flexible via `boundaries` and `fractions` parameters.

**Cons**: Sharp density transitions at boundaries may affect interpolation; requires choosing boundary location and point allocation.

### 3. Three Density Regions
Three-tier density: very dense near 0, medium density in middle, sparse in tails.

**Approach**: Pre-allocated density regions with fixed boundaries at 0.1W and 0.4W. Default allocation: 50% of points in inner 10% of range, 30% in next 30% of range, 20% in outer 60% of range.

**Pros**: Balances accuracy near ω=0 with coverage of full range; zero hyperparameters (uses sensible defaults); smooth density gradient; generally robust across problems.

**Cons**: Fixed default boundaries may not be optimal for all MDPs (though customizable via `boundaries` and `fractions` parameters).

### 4. Four Density Regions
Four-tier density: very dense near 0, progressively coarser toward tails.

**Approach**: Pre-allocated density regions with boundaries at 0.05W, 0.15W, and 0.4W. Default allocation: 40% of points in inner 5% of range, 30% in next 10%, 20% in next 25%, 10% in outer 60%.

**Pros**: Finest density control near ω=0; smooth multi-level density gradient; highly concentrated at origin where moment extraction is most critical.

**Cons**: More complex boundary structure; may over-concentrate near origin for some problems (though customizable via parameters).

### 5. Exponential Decay
Exponential density decay from origin: dense near 0, exponentially sparse in tails.

**Approach**: `ωₖ = ±W·(exp(k·λ/K_half) - 1)/(exp(λ) - 1)` with decay parameter λ. Use λ > 1 for concentration at center (default: λ=2.0).

**Pros**: Single tunable parameter λ controls concentration strength; smooth exponential density gradient (no sharp transitions); mathematically principled for decaying CFs; λ ≈ 1 approaches uniform, λ >> 1 heavily concentrates at 0.

**Cons**: Performance depends on CF decay rate; may allocate too few points in mid-range frequencies; requires tuning λ for optimal performance.

### 6. Linear Decay
Power-law density decay with linear exponent: moderate concentration near 0.

**Approach**: `ωₖ = ±W·(k/K_half)^α` with exponent α. Use α > 1 for concentration at center (default: α=2.0). Spacing grows linearly with distance from origin.

**Pros**: Single tunable parameter α controls concentration; α=1 gives uniform spacing, α>1 concentrates near 0; moderate concentration suitable for problems with gradual CF decay.

**Cons**: Less aggressive concentration than exponential or quadratic; may not allocate enough points near ω=0 for problems requiring high precision at origin.

### 7. Quadratic Decay
Power-law density decay with quadratic exponent: strong concentration near 0.

**Approach**: `ωₖ = ±W·(k/K_half)^β` with exponent β. Use β > 1 for concentration at center (default: β=3.0). Spacing grows quadratically with distance from origin.

**Pros**: Stronger concentration than linear decay; β=1 gives uniform, β>1 concentrates near 0; suitable for problems requiring high accuracy at ω=0; smooth density gradient.

**Cons**: Very aggressive concentration may over-allocate points near origin; may under-sample mid and tail regions; requires careful tuning of β.

---

## Empirical Performance

Comprehensive grid search experiments on Taxi-v3 (360 configurations testing all grid strategies) revealed a **striking finding**: grid strategy choice has minimal impact when paired with the right methods. The critical factor is **interpolation + collapse method combination**.

### Best Configurations (Top 40)
When using **`polar` interpolation + `gaussian` collapse**, ALL grid strategies achieve exceptional performance:
- **Four density regions**: MAE = 1.27×10⁻¹⁵ (best single config)
- **Linear decay** (K=128): MAE = 1.99×10⁻¹⁵
- **Quadratic decay** (K=128): MAE = 2.19×10⁻¹⁵
- **Three density regions**: MAE = 2.26×10⁻¹⁵ to 6.81×10⁻¹⁵
- **Two density regions**: MAE = 2.28×10⁻¹⁵ to 2.29×10⁻¹⁴
- **Exponential decay**: MAE = 2.99×10⁻¹⁵ to 6.08×10⁻¹⁴
- **Even uniform**: MAE = 3.18×10⁻¹⁴ to 5.22×10⁻¹⁴ (when properly configured)

All top configurations achieve MAE at or near **machine precision (10⁻¹⁵)**, demonstrating exact recovery of classical Value Iteration.

### Method Combination Impact
- **Polar + Gaussian**: 33 of top 40 configs (82.5%) - MAE ~10⁻¹⁵
- **Polar/PCHIP + LS**: Acceptable but orders of magnitude worse - MAE ~10⁻⁷
- **FFT collapse**: Catastrophically bad regardless of grid - MAE = 6 to 27
- **Lanczos + LS**: Consistently poor across all grids - MAE = 6 to 8

### Worst Configurations (Bottom 40)
The worst performers are determined by **method choice, not grid strategy**:
- **Uniform + FFT**: Absolute worst (MAE = 26.6)
- **Any grid + FFT**: Consistently terrible (dominates 27/40 worst configs)
- **Any grid + Lanczos + LS**: Another failure mode (13/40 worst configs)

Even advanced grid strategies (four density regions, exponential decay) fail catastrophically when paired with bad methods.

### Grid Size and Range
Surprisingly, **K and W have minimal impact** when using proper methods:
- K=128 with polar+gaussian: MAE ~10⁻¹⁵ (excellent)
- K=512 with FFT: MAE ~11 (catastrophic)
- Both W=10.0 and W=20.0 work equally well with proper methods

**Key Insight**: Grid strategy is secondary to method combination. Even a simple uniform grid achieves near-perfect results with `polar + gaussian`, while sophisticated multi-density grids fail with `FFT` or `lanczos + LS`.

**Recommendation**: 
1. **ALWAYS use** `polar` interpolation + `gaussian` collapse
2. **Any grid strategy works** with this combination - choose based on problem structure or use three/four density regions for slight edge
3. **NEVER use** FFT collapse or Lanczos+LS combination regardless of grid
4. K=256 is sufficient; even K=128 achieves excellent results with proper methods

