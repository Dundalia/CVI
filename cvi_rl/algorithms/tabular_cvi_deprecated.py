# cvi_rl/algorithms/tabular_cvi.py

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
from cvi_rl.cf.grids import make_omega_grid, GridStrategy


def compute_reward_cf_table(
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Compute reward characteristic functions per state under a fixed policy π.

    For each state s, we consider the one-step reward distribution R(s,π(s)):
      φ_R(s, ω) = E[exp(i ω R) | s, a=π(s)]

    For tabular MDPs with transition model P[s][a] = (prob, next_state, reward, done),
    this becomes:

      φ_R(s, ω) = sum_{(prob, *, r, *)} prob * exp(i ω r)

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment spec containing P.
    policy : np.ndarray
        Policy mapping state -> action, shape [n_states].
    omegas : np.ndarray
        Frequency grid, shape [K].

    Returns
    -------
    reward_cf_table : np.ndarray
        Complex array of shape [n_states, K], where reward_cf_table[s, k]
        is φ_R(s, ω_k).
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P
    K = len(omegas)

    # Basic sanity checks
    if policy.shape[0] != n_states:
        raise ValueError(
            f"Policy length {policy.shape[0]} does not match env states {n_states}."
        )

    reward_cf_table = np.zeros((n_states, K), dtype=complex)

    for s in range(n_states):
        a = int(policy[s])
        if not (0 <= a < n_actions):
            raise ValueError(f"Policy action {a} out of bounds for state {s}.")

        cf_s = np.zeros(K, dtype=complex)
        for prob, _, reward, _ in P[s][a]:
            # E[e^{i ω R}] = sum prob * e^{i ω r}
            cf_s += prob * np.exp(1j * omegas * reward)

        reward_cf_table[s] = cf_s

    return reward_cf_table


def cvi_policy_evaluation(
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    gamma: float = 0.9,
    grid_strategy: GridStrategy = "uniform",
    W: float = 8.0,
    K: int = 256,
    w_center: float = 3.0,
    frac_center: float = 0.9,
    eps: float = 1e-5,
    max_iters: int = 200,
    return_omegas: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Tabular Characteristic Value Iteration (CVI) for policy evaluation.

    This computes the fixed point V(s, ω) ≈ φ_{G}(s, ω) where G is the
    discounted return under policy π:

      G_π(s) = R_0 + γ R_1 + γ^2 R_2 + ...

    In the frequency domain, the Bellman equation becomes:

      V(s, ω) = φ_R(s, ω) * E[V(S', γ ω) | s, a=π(s)]

    We approximate this on a finite ω-grid, using linear interpolation
    to evaluate V(s, γ ω) between grid points.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment spec (n_states, n_actions, P, etc.).
    policy : np.ndarray
        Policy mapping state -> action, shape [n_states].
    gamma : float
        Discount factor.
    grid_strategy : {"uniform", "piecewise_centered"}
        How to discretize the frequency domain.
    W : float
        Max |ω| in the grid.
    K : int
        Number of frequency points.
    w_center : float
        For "piecewise_centered": half-width of dense central region.
    frac_center : float
        For "piecewise_centered": fraction of points in central region.
    eps : float
        Convergence threshold on max |V_{k+1} - V_k|.
    max_iters : int
        Maximum number of CVI iterations.
    return_omegas : bool
        If True, return the ω-grid alongside the CVI solution.

    Returns
    -------
    V : np.ndarray
        Complex array of shape [n_states, K], representing V(s, ω_k)
        for all states and frequencies.
    omegas : np.ndarray or None
        The ω-grid, shape [K], if return_omegas=True, else None.

    Example
    -------
    >>> from cvi_rl.envs.registry import make_env
    >>> from cvi_rl.algorithms.tabular_cvi import cvi_policy_evaluation
    >>> env_spec, env = make_env("taxi")
    >>> random_policy = np.random.randint(env_spec.n_actions, size=env_spec.n_states)
    >>> V_cvi, omegas = cvi_policy_evaluation(
    ...     env_spec, random_policy, gamma=0.9,
    ...     grid_strategy="piecewise_centered", W=8.0, K=256,
    ...     w_center=3.0, frac_center=0.9,
    ... )
    """
    n_states = env_spec.n_states
    P: TransitionModel = env_spec.P

    # 1) Build ω-grid according to chosen strategy
    omegas = make_omega_grid(
        strategy=grid_strategy,
        W=W,
        K=K,
        w_center=w_center,
        frac_center=frac_center,
    )

    K = len(omegas)
    scaled_omegas = gamma * omegas

    # 2) Precompute reward CF φ_R(s, ω)
    reward_cf_table = compute_reward_cf_table(env_spec, policy, omegas)

    # 3) Initialize V(s, ω) = 1 (CF of zero-return RV)
    V = np.ones((n_states, K), dtype=complex)

    for _ in range(max_iters):
        V_prev = V.copy()

        # 3a) Compute V_prev(s, γ ω) via interpolation
        V_scaled = np.zeros_like(V_prev)

        for s in range(n_states):
            real_scaled = np.interp(
                scaled_omegas,
                omegas,
                V_prev[s].real,
                left=V_prev[s].real[0],
                right=V_prev[s].real[-1],
            )
            imag_scaled = np.interp(
                scaled_omegas,
                omegas,
                V_prev[s].imag,
                left=V_prev[s].imag[0],
                right=V_prev[s].imag[-1],
            )
            V_scaled[s] = real_scaled + 1j * imag_scaled

        # 3b) Bellman update for each state
        max_delta = 0.0

        for s in range(n_states):
            a = int(policy[s])

            # Expectation over next states under P(s,a)
            exp_next = np.zeros(K, dtype=complex)
            for prob, next_state, _, done in P[s][a]:
                if done:
                    # Terminal: future return is 0 so CF is 1
                    exp_next += prob * 1.0
                else:
                    exp_next += prob * V_scaled[next_state]

            V[s] = reward_cf_table[s] * exp_next

            max_delta = max(max_delta, np.max(np.abs(V[s] - V_prev[s])))

        if max_delta < eps:
            break

    if return_omegas:
        return V, omegas

    return V, None