# cvi_rl/cf/grids.py

from __future__ import annotations

from typing import Literal

import numpy as np

GridStrategy = Literal["uniform", "piecewise_centered"]


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


def make_omega_grid(
    strategy: GridStrategy = "uniform",
    W: float = 8.0,
    K: int = 256,
    w_center: float = 3.0,
    frac_center: float = 0.9,
) -> np.ndarray:
    """
    Factory for ω-grids, selecting a strategy by string.

    Parameters
    ----------
    strategy : {"uniform", "piecewise_centered"}
        Which grid construction to use.
    W : float
        Overall max frequency |ω|.
    K : int
        Number of grid points.
    w_center : float
        Only used by "piecewise_centered": half-width of dense region.
    frac_center : float
        Only used by "piecewise_centered": fraction of points in dense region.

    Returns
    -------
    omegas : np.ndarray
        1D array of ω-values of shape [K].

    Examples
    --------
    >>> omegas = make_omega_grid(strategy="uniform", W=8.0, K=256)
    >>> omegas = make_omega_grid(strategy="piecewise_centered",
    ...                          W=8.0, K=256, w_center=3.0, frac_center=0.9)
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
    else:
        raise ValueError(
            f"Unknown ω-grid strategy: {strategy!r}. "
            f"Supported: 'uniform', 'piecewise_centered'."
        )