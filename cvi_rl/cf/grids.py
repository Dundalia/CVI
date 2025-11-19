# cvi_rl/cf/grids.py

from __future__ import annotations

from typing import Literal

import numpy as np

GridStrategy = Literal["uniform", "piecewise_centered", "logarithmic", "chebyshev", "adaptive"]


def make_uniform_grid(W: float, K: int) -> np.ndarray:
    """
    Uniform frequency grid in [-W, W].

    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points.

    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], linearly spaced between -W and W.
    """
    return np.linspace(-W, W, K)


def make_piecewise_centered_grid(
    W: float,
    K: int,
    w_center: float,
    frac_center: float,
) -> np.ndarray:
    """
    Piecewise non-uniform grid, denser near 0:

      - A dense "center" region in [-w_center, w_center]
      - Two coarser "tail" regions in [-W, -w_center) and (w_center, W]

    Parameters
    ----------
    W : float
        Overall max frequency range (grid in [-W, W]).
    K : int
        Total number of points.
    w_center : float
        Half-width of the dense central region.
    frac_center : float
        Fraction of grid points assigned to the central region (0 < frac_center < 1).

    Returns
    -------
    omegas : np.ndarray
        Sorted 1D array of shape [K].
    """
    if not (0.0 < frac_center < 1.0):
        raise ValueError(f"frac_center must be in (0,1), got {frac_center}")

    K_center = int(frac_center * K)
    K_center = max(3, min(K - 2, K_center))  # ensure room for tails

    K_tail_each = max(1, (K - K_center) // 2)

    # Center region (dense)
    omegas_center = np.linspace(-w_center, w_center, K_center, endpoint=True)

    # Tails (coarser), avoid duplicating endpoints
    omegas_left = np.linspace(-W, -w_center, K_tail_each, endpoint=False)
    omegas_right = np.linspace(w_center, W, K_tail_each, endpoint=False)

    omegas = np.concatenate([omegas_left, omegas_center, omegas_right])

    # Adjust length to exactly K (clip or pad if needed)
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        pad_needed = K - len(omegas)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left

        left_pad = np.linspace(-W, omegas[0], pad_left + 1, endpoint=False)
        right_pad = np.linspace(omegas[-1], W, pad_right + 1, endpoint=True)[1:]

        omegas = np.concatenate([left_pad, omegas, right_pad])

    omegas = np.sort(omegas)
    return omegas


def make_logarithmic_grid(
    W: float,
    K: int,
    lam: float = 2.0,
) -> np.ndarray:
    """
    Logarithmic spacing from origin, dense near 0 and sparse in tails.
    
    Uses exponential spacing: ω = ±W · (exp(k·λ/K_half) - 1) / (exp(λ) - 1)
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points (should be even for symmetry).
    lam : float
        Decay rate parameter. Higher λ → more points near 0.
        λ=0 gives uniform, λ→∞ gives all points at 0.
        Default: 2.0 (good balance).
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], logarithmically spaced in [-W, W].
    """
    K_half = K // 2
    
    if lam < 1e-6:
        # Fallback to uniform for very small lambda
        return np.linspace(-W, W, K)
    
    # Create positive half using exponential spacing
    # k ranges from 0 to K_half
    k = np.arange(K_half + 1)
    
    # Exponential transform: maps [0, K_half] → [0, W]
    # with higher density near 0
    scale = (np.exp(k * lam / K_half) - 1) / (np.exp(lam) - 1)
    omegas_pos = W * scale
    
    # Mirror for negative half (exclude 0 if already present)
    if K % 2 == 0:
        # Even K: no point exactly at 0
        omegas_neg = -omegas_pos[1:][::-1]
        omegas = np.concatenate([omegas_neg, omegas_pos])
    else:
        # Odd K: include 0
        omegas_neg = -omegas_pos[1:][::-1]
        omegas = np.concatenate([omegas_neg, omegas_pos])
    
    # Ensure exactly K points
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        # Pad with endpoint if needed
        omegas = np.append(omegas, W)[:K]
    
    return np.sort(omegas)


def make_chebyshev_grid(
    W: float,
    K: int,
) -> np.ndarray:
    """
    Chebyshev nodes (Chebyshev points of the first kind) in [-W, W].
    
    These are the zeros of the Chebyshev polynomial T_K(x).
    Minimizes Runge phenomenon in polynomial interpolation.
    Naturally denser near boundaries ±W and also relatively dense near 0.
    
    Formula: ω_k = W · cos(π(2k-1)/(2K)) for k = 1,...,K
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points.
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], Chebyshev nodes in [-W, W].
    """
    k = np.arange(1, K + 1)
    # Chebyshev nodes in [-1, 1]
    nodes = np.cos(np.pi * (2 * k - 1) / (2 * K))
    # Scale to [-W, W]
    omegas = W * nodes
    return np.sort(omegas)


def make_adaptive_grid(
    W: float,
    K: int,
    refinement_regions: int = 3,
) -> np.ndarray:
    """
    Adaptive grid with multiple density regions.
    
    This is a simplified adaptive grid that pre-allocates points based on
    expected CF behavior. For true adaptive refinement, this would need
    to be called iteratively during CVI with curvature feedback.
    
    The grid has three regions with different densities:
    - Very dense near 0 (where mean extraction happens)
    - Medium dense in middle region (where CF transitions)
    - Sparse in tails (where CF decays/oscillates rapidly)
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points.
    refinement_regions : int
        Number of density regions (default: 3).
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], adaptively spaced in [-W, W].
    """
    # Allocate points across regions with decreasing density
    # Region 1: [-w1, w1] - very dense (50% of points in 20% of range)
    # Region 2: [-w2, -w1] and [w1, w2] - medium (30% of points in 30% of range)  
    # Region 3: [-W, -w2] and [w2, W] - sparse (20% of points in 50% of range)
    
    w1 = W * 0.1  # Inner boundary (10% of range)
    w2 = W * 0.4  # Outer boundary (40% of range)
    
    # Allocate points
    K1 = int(0.5 * K)  # Very dense near 0
    K2 = int(0.3 * K)  # Medium in middle
    K3 = K - K1 - K2   # Sparse in tails
    
    # Ensure minimum points in each region
    K1 = max(3, K1)
    K2 = max(2, K2)
    K3 = max(2, K3)
    
    # Build grid segments
    # Center: [-w1, w1]
    center = np.linspace(-w1, w1, K1)
    
    # Middle regions: [-w2, -w1] and [w1, w2]
    K2_half = K2 // 2
    middle_left = np.linspace(-w2, -w1, K2_half, endpoint=False)
    middle_right = np.linspace(w1, w2, K2_half, endpoint=False)
    
    # Tail regions: [-W, -w2] and [w2, W]
    K3_half = K3 // 2
    tail_left = np.linspace(-W, -w2, K3_half, endpoint=False)
    tail_right = np.linspace(w2, W, K3 - K3_half, endpoint=True)
    
    # Combine all regions
    omegas = np.concatenate([tail_left, middle_left, center, middle_right, tail_right])
    
    # Adjust to exactly K points
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        # Pad uniformly in the largest gap
        omegas = np.sort(omegas)
        gaps = np.diff(omegas)
        max_gap_idx = np.argmax(gaps)
        insert_point = (omegas[max_gap_idx] + omegas[max_gap_idx + 1]) / 2
        omegas = np.sort(np.append(omegas, insert_point))[:K]
    
    return np.sort(omegas)


def make_omega_grid(
    strategy: GridStrategy = "uniform",
    W: float = 8.0,
    K: int = 256,
    w_center: float = 3.0,
    frac_center: float = 0.9,
    lam: float = 2.0,
    refinement_regions: int = 3,
) -> np.ndarray:
    """
    Factory for ω-grids, selecting a strategy by string.

    Parameters
    ----------
    strategy : {"uniform", "piecewise_centered", "logarithmic", "chebyshev", "adaptive"}
        Which grid construction to use.
    W : float
        Overall max frequency |ω|.
    K : int
        Number of grid points.
    w_center : float
        Only used by "piecewise_centered": half-width of dense region.
    frac_center : float
        Only used by "piecewise_centered": fraction of points in dense region.
    lam : float
        Only used by "logarithmic": decay rate parameter (default: 2.0).
        Higher λ → more points near 0.
    refinement_regions : int
        Only used by "adaptive": number of density regions (default: 3).

    Returns
    -------
    omegas : np.ndarray
        1D array of ω-values of shape [K].

    Examples
    --------
    >>> omegas = make_omega_grid(strategy="uniform", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="piecewise_centered",
    ...                          W=8.0, K=256, w_center=3.0, frac_center=0.9)
    >>> omegas = make_omega_grid(strategy="logarithmic", W=8.0, K=256, lam=2.0)
    >>> omegas = make_omega_grid(strategy="chebyshev", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="adaptive", W=8.0, K=256)
    """
    if strategy == "uniform":
        return make_uniform_grid(W=W, K=K)
    elif strategy == "piecewise_centered":
        return make_piecewise_centered_grid(
            W=W,
            K=K,
            w_center=w_center,
            frac_center=frac_center,
        )
    elif strategy == "logarithmic":
        return make_logarithmic_grid(W=W, K=K, lam=lam)
    elif strategy == "chebyshev":
        return make_chebyshev_grid(W=W, K=K)
    elif strategy == "adaptive":
        return make_adaptive_grid(W=W, K=K, refinement_regions=refinement_regions)
    else:
        raise ValueError(
            f"Unknown ω-grid strategy: {strategy!r}. "
            f"Supported: 'uniform', 'piecewise_centered', 'logarithmic', 'chebyshev', 'adaptive'."
        )