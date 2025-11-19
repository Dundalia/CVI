## **CVI (Characteristic Value Iteration) — Current Contributions & Design Choices**

### **Contributions Implemented So Far**
1. **Tabular CVI Policy Evaluation**
   - Implemented the CF Bellman operator on a discretized ω-grid.
   - Computes \(V_\pi(s,\omega)\) for any fixed policy on any tabular MDP.

2. **CF-Based Action Evaluation**
   - Given \(V(s,\omega)\), computes \(Q_{\text{CF}}(s,a,\omega)\) via:
     \[
     Q_{\text{CF}}(s,a,\omega) = \mathbb{E}\left[e^{i\omega R} \cdot V(s',\gamma\omega)\right].
     \]

3. **Scalar Collapse Operator**
   - Derives scalar \(Q(s,a)\) by estimating the **mean return** from the CF:
     - LS-based local quadratic fit around ω = 0 to approximate φ′(0).
     - Proven to recover the **optimal VI policy** in at least one tested setup.

4. **Monte Carlo CF Reference**
   - MC-based empirical CF:
     \[
     \varphi_{\text{MC}}(\omega) = \mathbb{E}[e^{i\omega G}]
     \]
   - Allows CF-MSE evaluation for tuning and diagnostics.

5. **Multiple ω-grid Strategies**
   - **Uniform grid**: Evenly spaced points in \([-W, W]\).
   - **Piecewise-centered grid**: Dense sampling near \(|\omega| ≈ 0\), coarser in tails.
     - Controlled by `w_center` and `frac_center` parameters.
     - Empirically superior for mean extraction (3x lower error than uniform).

6. **CVI Action-Control Prototype**
   - Compute \(Q_{\text{CF}}\) → collapse → greedy policy:
     \[
     \pi(s) = \arg\max_a Q(s,a)
     \]
   - Shown to produce the **same greedy policy as classical VI** under some conditions.

---

### **Design Choices Implemented**
#### **1. ω-grid Construction**
- **Uniform**: evenly spaced in \([-W, W]\).
- **Piecewise-centered**: high density near \(|\omega| \approx 0\).
- Hyperparameters:
  - `W` (frequency range)
  - `K` (number of points)
  - `w_center`, `frac_center`, `strategy` string selector.

#### **2. Interpolation Methods**
Multiple interpolation schemes for evaluating \(V(s, \gamma\omega)\) between grid points:

- **Linear (Cartesian)**: Standard linear interpolation of real and imaginary parts separately.
  - Fast and stable, works well for most cases.
  
- **Polar**: Interpolates magnitude \(|φ|\) and phase \(\arg(φ)\) separately with phase unwrapping.
  - Better preserves CF validity constraint \(|φ(ω)| ≤ 1\).
  - Empirically performs best in grid search experiments.
  
- **PCHIP (Piecewise Cubic Hermite)**: Monotonicity-preserving cubic interpolation.
  - Avoids overshoots common in cubic splines.
  - Applied to real/imaginary parts independently.
  
- **Lanczos**: Windowed sinc interpolation (windowed by sinc(x/a)).
  - Theoretically optimal for bandlimited signals on uniform grids.
  - Computationally expensive; best for uniform grids only.

#### **3. CF Moment Extraction / Collapse Methods**
Multiple methods for extracting the mean \(E[G] = \text{Im}(φ'(0))\) from the characteristic function:

- **LS (Least Squares)**: Local quadratic fit around ω=0.
  - Fits \(φ(ω) ≈ a_0 + a_1 ω + a_2 ω^2\) using points in window \([−m, m]\).
  - Extracts mean from \(a_1\): \(E[G] = \text{Im}(a_1)\).
  - Can also estimate variance from \(a_2\): \(\text{Var}[G] ≈ -\text{Re}(a_2)\).
  - Robust and consistent, moderate accuracy.
  
- **FFT (Fast Fourier Transform)**: Inverse Fourier transform to spatial domain.
  - Computes PDF/PMF via \(\text{IFFT}(φ)\), then calculates mean directly.
  - Requires uniform grids; sensitive to aliasing and normalization.
  - Empirically least reliable (large errors in experiments).
  
- **Gaussian**: Phase unwrapping with linear regression.
  - Assumes locally Gaussian CF: \(\log φ(ω) ≈ iμω - \frac{1}{2}σ^2ω^2\).
  - Unwraps phase \(\arg(φ(ω))\) and fits line: \(\text{phase} ≈ μω\).
  - Most accurate in experiments; works best with smooth CFs.
  
- **Savitzky-Golay**: Smoothing filter with derivative estimation.
  - Applies SG filter to \(\text{Im}(φ(ω))\) with \(\text{deriv}=1\).
  - Evaluates derivative at ω=0 to estimate mean.
  - Configurable window length and polynomial order.

#### **4. Collapse Operator for Control**
- Scalar \(Q(s,a)\) obtained from **mean-based collapse**:
  \[
  Q(s,a) = \text{Im}(\varphi'_{Q}(0))
  \]
- Enables greedy control in CF space.

#### **5. CVI Convergence Settings**
- `eps` tolerance on sup-norm over (s, ω).
- Max iteration cap.
- Initialization at φ(ω)=1 (return=0 baseline).

#### **6. Terminal State Handling**
- Terminal states contribute multiplicatively with φ(0)=1.
- Supports stochastic or deterministic terminations in P.

#### **7. Evaluation Metrics**
- CF-MSE vs MC CF.
- Mean/variance errors.
- Policy agreement vs classical VI.
- Runtime and stability tracking.

### **Additional Notes**

#### **Evaluation Modes**
We support **two evaluation modes**:  
  **(i)** frequency-space evaluation via CF-MSE, comparing the CVI-predicted CF against the **Monte-Carlo empirical characteristic function (ECF)**  
  \(\hat{\varphi}_{\text{MC}}(\omega)=\frac{1}{N}\sum_k e^{i\omega G_k}\);  
  **(ii)** state-space evaluation by collapsing CF → scalar values and comparing against VI or MC.  
  State-space evaluation is generally **more stable and RL-relevant**.

#### **CF Validity Constraints**
Our CVI operator currently does **not enforce CF validity constraints**, such as  
  φ(0)=1,   |φ(ω)|≤1,   and Hermitian symmetry φ(−ω)=φ(ω)\*.  
  Future versions may incorporate **CF-validity projections or regularizers**.

#### **Grid Search Experiments**
A comprehensive hyperparameter grid search was conducted to evaluate the performance of different CVI configurations. The complete experimental setup and results are available in:

**Notebook**: `cvi_rl/experiment_suite.ipynb`

**Experimental Design**:
- **Environment**: Taxi-v3 (500 states, 6 actions)
- **Baseline**: Classical Value Iteration with γ=0.9
- **Configurations tested**: 264 successful combinations (after removing Savgol, fixing PCHIP) spanning:
  - Grid strategies: uniform, piecewise-centered, logarithmic, chebyshev, adaptive
  - Frequency ranges (W): 10.0, 20.0
  - Grid sizes (K): 128, 256, 512
  - Interpolation methods: linear, polar, pchip, lanczos
  - Collapse methods: ls, fft, gaussian
- **Evaluation metric**: Mean Absolute Error (MAE) between CVI-derived Q-values and classical VI Q-values
- **Goal**: Identify which combinations of methods allow CVI to exactly recover the optimal policy

**Key Findings**:
Grid search experiments (264 successful configurations on Taxi-v3) revealed:
- **Best configuration**: adaptive/piecewise-centered grid + polar interpolation + gaussian collapse
  - Achieves near-zero error (MAE ≈ 10⁻¹⁵, exact match with VI)
- **Grid strategy impact**: Adaptive (1.61) > Piecewise-centered (1.76) > Logarithmic (1.97) > Chebyshev (4.98) > Uniform (6.09)
  - **Adaptive is the clear winner** with zero hyperparameters - automatically concentrates density near ω=0
  - Appears in 20/75 excellent configs (most of any strategy)
  - Chebyshev improved after fixes but still poor (wrong density distribution for CVI)
- **Interpolation impact**: Linear (2.93) ≈ Polar (2.90) ≈ PCHIP (2.98) << Lanczos (5.34)
  - Surprise: Simple methods work just as well as sophisticated ones
  - Polar appears in all top 5 configs (best peak performance)
  - Lanczos significantly underperforms despite theoretical advantages
- **Collapse method impact**: Gaussian (2.08) >> LS (3.34) >> FFT (11.81)
  - Gaussian dominates excellent configs: 72/75 (96%)
- **Grid size impact**: Larger K consistently better: K=512 (2.50) > K=256 (3.30) > K=128 (4.81)
  - Previous concerns about larger K were artifacts of buggy methods (Savgol, pre-fix PCHIP)