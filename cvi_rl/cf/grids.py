# cvi_rl/cf/grids.py

from __future__ import annotations

from typing import Literal, List

import numpy as np

GridStrategy = Literal[
    "uniform",
    "two_density_regions",
    "three_density_regions",
    "four_density_regions",
    "exponential_decay",
    "linear_decay",
    "quadratic_decay",
]


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


def make_multi_density_regions_grid(
    W: float,
    K: int,
    boundaries: List[float],
    fractions: List[float],
) -> np.ndarray:
    """
    Multi-region grid with configurable density levels from center to tails.
    
    Creates a symmetric grid around 0 with N density regions, where N = len(fractions).
    The innermost region is the densest, with decreasing density toward the tails.
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency (grid spans [-W, W]).
    K : int
        Total number of grid points.
    boundaries : List[float]
        Relative boundary positions as fractions of W, from inner to outer.
        For N regions, provide N-1 boundaries.
        Example: [0.1, 0.4] means boundaries at ±0.1W and ±0.4W (3 regions).
    fractions : List[float]
        Fraction of total points K allocated to each region, from inner to outer.
        Must sum to 1.0 and have length = len(boundaries) + 1.
        Example: [0.5, 0.3, 0.2] allocates 50% to center, 30% to middle, 20% to tails.
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with multi-density regions in [-W, W].
    
    Examples
    --------
    >>> # Two regions: dense center (90%) + sparse tails (10%)
    >>> omegas = make_multi_density_regions_grid(W=10.0, K=256, 
    ...     boundaries=[0.3], fractions=[0.9, 0.1])
    >>> 
    >>> # Three regions: 50% center, 30% middle, 20% tails
    >>> omegas = make_multi_density_regions_grid(W=10.0, K=256,
    ...     boundaries=[0.1, 0.4], fractions=[0.5, 0.3, 0.2])
    """
    n_regions = len(fractions)
    
    # Validate inputs
    if len(boundaries) != n_regions - 1:
        raise ValueError(
            f"Expected {n_regions - 1} boundaries for {n_regions} regions, "
            f"got {len(boundaries)}"
        )
    
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fractions)}")
    
    # Convert relative boundaries to absolute values
    abs_boundaries = [W * b for b in boundaries]
    
    # Build region edges: [0, w1, w2, ..., W]
    region_edges = [0.0] + abs_boundaries + [W]
    
    # Allocate points to each region
    K_per_region = []
    K_remaining = K
    for i, frac in enumerate(fractions[:-1]):
        k = max(2 if i > 0 else 3, int(frac * K))  # min 3 for center, 2 for others
        K_per_region.append(k)
        K_remaining -= k
    K_per_region.append(max(2, K_remaining))  # Last region gets remainder
    
    # Build grid segments for each region
    segments = []
    
    for i in range(n_regions):
        inner_edge = region_edges[i]
        outer_edge = region_edges[i + 1]
        k_region = K_per_region[i]
        
        if i == 0:
            # Center region: symmetric around 0, includes both endpoints
            center = np.linspace(-inner_edge if inner_edge > 0 else -outer_edge, 
                                  inner_edge if inner_edge > 0 else outer_edge, 
                                  k_region)
            # For center region, we use the outer edge as the boundary
            center = np.linspace(-outer_edge, outer_edge, k_region)
            segments.append(center)
        else:
            # Non-center regions: split into left and right halves
            k_half = k_region // 2
            
            # Left side: [-outer_edge, -inner_edge)
            left = np.linspace(-outer_edge, -inner_edge, k_half, endpoint=False)
            
            # Right side: [inner_edge, outer_edge) or [inner_edge, outer_edge] for last
            if i == n_regions - 1:
                # Last region includes the endpoint W
                right = np.linspace(inner_edge, outer_edge, k_region - k_half, endpoint=True)
            else:
                right = np.linspace(inner_edge, outer_edge, k_region - k_half, endpoint=False)
            
            segments.insert(0, left)  # Add left to beginning
            segments.append(right)    # Add right to end
    
    # Combine all segments
    omegas = np.concatenate(segments)
    
    # Adjust to exactly K points if needed
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        # Pad uniformly in the largest gap
        omegas = np.sort(omegas)
        while len(omegas) < K:
            gaps = np.diff(omegas)
            max_gap_idx = np.argmax(gaps)
            insert_point = (omegas[max_gap_idx] + omegas[max_gap_idx + 1]) / 2
            omegas = np.sort(np.append(omegas, insert_point))
        omegas = omegas[:K]
    
    # Ensure strictly increasing (remove duplicates)
    omegas = np.unique(omegas)
    return omegas


def make_two_density_regions_grid(
    W: float,
    K: int,
    boundaries: List[float] = None,
    fractions: List[float] = None,
) -> np.ndarray:
    """
    Two-region grid: dense center + sparse tails.
    
    Default allocation:
    - Center (90% of points in inner 30% of range)
    - Tails (10% of points in outer 70% of range)
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency (grid spans [-W, W]).
    K : int
        Total number of grid points.
    boundaries : List[float], optional
        Relative boundary position [center_edge] as fraction of W.
        Default: [0.3].
    fractions : List[float], optional
        Point allocation fractions [center, tails].
        Default: [0.9, 0.1].
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with two density regions in [-W, W].
    """
    if boundaries is None:
        boundaries = [0.3]
    if fractions is None:
        fractions = [0.9, 0.1]
    
    return make_multi_density_regions_grid(
        W=W,
        K=K,
        boundaries=boundaries,
        fractions=fractions,
    )


def make_three_density_regions_grid(
    W: float,
    K: int,
    boundaries: List[float] = None,
    fractions: List[float] = None,
) -> np.ndarray:
    """
    Three-region grid: very dense center, medium middle, sparse tails.
    
    Default allocation:
    - Center (50% of points in inner 10% of range)
    - Middle (30% of points in next 30% of range)  
    - Tails (20% of points in outer 60% of range)
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency (grid spans [-W, W]).
    K : int
        Total number of grid points.
    boundaries : List[float], optional
        Relative boundary positions [inner, outer] as fractions of W.
        Default: [0.1, 0.4].
    fractions : List[float], optional
        Point allocation fractions [center, middle, tails].
        Default: [0.5, 0.3, 0.2].
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with three density regions in [-W, W].
    """
    if boundaries is None:
        boundaries = [0.1, 0.4]
    if fractions is None:
        fractions = [0.5, 0.3, 0.2]
    
    return make_multi_density_regions_grid(
        W=W,
        K=K,
        boundaries=boundaries,
        fractions=fractions,
    )


def make_four_density_regions_grid(
    W: float,
    K: int,
    boundaries: List[float] = None,
    fractions: List[float] = None,
) -> np.ndarray:
    """
    Four-region grid: very dense center, dense inner-mid, medium outer-mid, sparse tails.
    
    Default allocation:
    - Center (40% of points in inner 5% of range)
    - Inner-middle (30% of points in next 10% of range)
    - Outer-middle (20% of points in next 25% of range)
    - Tails (10% of points in outer 60% of range)
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency (grid spans [-W, W]).
    K : int
        Total number of grid points.
    boundaries : List[float], optional
        Relative boundary positions [inner, mid, outer] as fractions of W.
        Default: [0.05, 0.15, 0.4].
    fractions : List[float], optional
        Point allocation fractions [center, inner-mid, outer-mid, tails].
        Default: [0.4, 0.3, 0.2, 0.1].
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with four density regions in [-W, W].
    """
    if boundaries is None:
        boundaries = [0.05, 0.15, 0.4]
    if fractions is None:
        fractions = [0.4, 0.3, 0.2, 0.1]
    
    return make_multi_density_regions_grid(
        W=W,
        K=K,
        boundaries=boundaries,
        fractions=fractions,
    )


def make_exponential_decay_grid(
    W: float,
    K: int,
    lam: float = 2.0,
) -> np.ndarray:
    """
    Exponential density decay from origin, dense near 0 and sparse in tails.
    
    Uses exponential spacing: ω = ±W · (exp(k·λ/K_half) - 1) / (exp(λ) - 1)
    
    The exponent λ controls concentration at the center. Use λ > 1 for 
    concentration near ω=0 (higher values = stronger concentration).
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points (should be even for symmetry).
    lam : float
        Exponential decay rate parameter. Must be > 1 for concentration at center.
        Higher λ → more points near 0 (stronger concentration).
        λ ≈ 1 approaches uniform, λ >> 1 concentrates heavily at 0.
        Default: 2.0 (good balance).
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with exponential density decay in [-W, W].
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


def make_linear_decay_grid(
    W: float,
    K: int,
    alpha: float = 2.0,
) -> np.ndarray:
    """
    Power-law density decay from origin, dense near 0 and sparser in tails.
    
    Uses power-law spacing with exponent α: ω = ±W · (k/K_half)^α
    
    The exponent α controls concentration at the center. Use α > 1 for 
    concentration near ω=0 (higher values = stronger concentration).
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points (should be even for symmetry).
    alpha : float
        Power-law exponent. Must be > 1 for concentration at center.
        Higher α → more points near 0 (stronger concentration).
        α = 1 gives uniform spacing, α >> 1 concentrates heavily at 0.
        Default: 2.0.
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with power-law density decay in [-W, W].
    """
    K_half = K // 2
    
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    
    if abs(alpha - 1.0) < 1e-6:
        # alpha=1 gives uniform spacing
        return np.linspace(-W, W, K)
    
    # Create positive half using power-law spacing
    # k ranges from 0 to K_half
    k = np.arange(K_half + 1)
    
    # Power-law transform: maps [0, K_half] → [0, W]
    # with higher density near 0 when alpha > 1
    scale = (k / K_half) ** alpha
    omegas_pos = W * scale
    
    # Mirror for negative half
    omegas_neg = -omegas_pos[1:][::-1]
    omegas = np.concatenate([omegas_neg, omegas_pos])
    
    # Ensure exactly K points
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        omegas = np.append(omegas, W)[:K]
    
    return np.sort(omegas)


def make_quadratic_decay_grid(
    W: float,
    K: int,
    beta: float = 3.0,
) -> np.ndarray:
    """
    Strong power-law density decay from origin, very dense near 0 and sparser in tails.
    
    Uses power-law spacing with higher exponent: ω = ±W · (k/K_half)^β
    
    The exponent β controls concentration at the center. Use β > 1 for 
    concentration near ω=0 (higher values = stronger concentration).
    Typically β > α (from linear_decay) for stronger concentration.
    
    Parameters
    ----------
    W : float
        Maximum absolute frequency.
    K : int
        Number of points (should be even for symmetry).
    beta : float
        Power-law exponent. Must be > 1 for concentration at center.
        Higher β → more points near 0 (stronger concentration).
        β = 1 gives uniform spacing, β >> 1 concentrates heavily at 0.
        Default: 3.0 (stronger concentration than linear_decay's α=2.0).
    
    Returns
    -------
    omegas : np.ndarray
        1D array of shape [K], with strong power-law density decay in [-W, W].
    """
    K_half = K // 2
    
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    
    if abs(beta - 1.0) < 1e-6:
        # beta=1 gives uniform spacing
        return np.linspace(-W, W, K)
    
    # Create positive half using power-law spacing with higher exponent
    # k ranges from 0 to K_half
    k = np.arange(K_half + 1)
    
    # Power-law transform with higher exponent for stronger concentration
    scale = (k / K_half) ** beta
    omegas_pos = W * scale
    
    # Mirror for negative half
    omegas_neg = -omegas_pos[1:][::-1]
    omegas = np.concatenate([omegas_neg, omegas_pos])
    
    # Ensure exactly K points
    if len(omegas) > K:
        omegas = omegas[:K]
    elif len(omegas) < K:
        omegas = np.append(omegas, W)[:K]
    
    return np.sort(omegas)


def make_omega_grid(
    strategy: GridStrategy = "uniform",
    W: float = 8.0,
    K: int = 256,
    lam: float = 2.0,
    alpha: float = 2.0,
    beta: float = 3.0,
) -> np.ndarray:
    """
    Factory for ω-grids, selecting a strategy by string.

    Parameters
    ----------
    strategy : GridStrategy
        Which grid construction to use. Options:
        - "uniform": evenly spaced points
        - "two_density_regions": dense center + sparse tails (2 densities)
        - "three_density_regions": 3 density levels from center to tails
        - "four_density_regions": 4 density levels from center to tails
        - "exponential_decay": exponential spacing, dense near 0 (use λ > 1)
        - "linear_decay": power-law spacing with exponent α (use α > 1)
        - "quadratic_decay": power-law spacing with exponent β (use β > 1)
    W : float
        Overall max frequency |ω|.
    K : int
        Number of grid points.
    lam : float
        Only used by "exponential_decay": decay rate parameter (default: 2.0).
        Must be > 1 for concentration at center. Higher λ → more points near 0.
    alpha : float
        Only used by "linear_decay": concentration parameter (default: 2.0).
        Must be > 1 for concentration at center. Higher α → more points near 0.
    beta : float
        Only used by "quadratic_decay": concentration parameter (default: 3.0).
        Must be > 1 for concentration at center. Higher β → more points near 0.

    Returns
    -------
    omegas : np.ndarray
        1D array of ω-values of shape [K].

    Examples
    --------
    >>> omegas = make_omega_grid(strategy="uniform", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="two_density_regions", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="three_density_regions", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="four_density_regions", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="exponential_decay", W=8.0, K=256, lam=2.0)
    >>> omegas = make_omega_grid(strategy="linear_decay", W=8.0, K=256, alpha=2.0)
    >>> omegas = make_omega_grid(strategy="quadratic_decay", W=8.0, K=256, beta=3.0)
    """
    if strategy == "uniform":
        return make_uniform_grid(W=W, K=K)
    elif strategy == "two_density_regions":
        return make_two_density_regions_grid(W=W, K=K)
    elif strategy == "three_density_regions":
        return make_three_density_regions_grid(W=W, K=K)
    elif strategy == "four_density_regions":
        return make_four_density_regions_grid(W=W, K=K)
    elif strategy == "exponential_decay":
        return make_exponential_decay_grid(W=W, K=K, lam=lam)
    elif strategy == "linear_decay":
        return make_linear_decay_grid(W=W, K=K, alpha=alpha)
    elif strategy == "quadratic_decay":
        return make_quadratic_decay_grid(W=W, K=K, beta=beta)
    else:
        raise ValueError(
            f"Unknown ω-grid strategy: {strategy!r}. "
            f"Supported: 'uniform', 'two_density_regions', 'three_density_regions', "
            f"'four_density_regions', 'exponential_decay', 'linear_decay', 'quadratic_decay'."
        )
