# cvi_rl/algorithms/tabular_cvi.py

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
from cvi_rl.cf.grids import make_omega_grid, GridStrategy
from cvi_rl.cf.processing import (
    CollapseMethod,
    InterpolationMethod,
    estimate_mean_ls,
    estimate_mean_fft,
    estimate_mean_gaussian,
    estimate_mean_savgol,
    interpolate_linear,
    interpolate_polar,
    interpolate_pchip,
    interpolate_lanczos
)


# ---------------------------------------------------------------------------
# Reward CFs
# ---------------------------------------------------------------------------

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
            cf_s += prob * np.exp(1j * omegas * reward)

        reward_cf_table[s] = cf_s

    return reward_cf_table


def compute_reward_cf_table_sa(
    env_spec: TabularEnvSpec,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Compute reward characteristic functions for every state-action pair:

      φ_R(s,a,ω) = E[exp(i ω R) | s, a]

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment spec containing P.
    omegas : np.ndarray
        Frequency grid, shape [K].

    Returns
    -------
    reward_cf_sa : np.ndarray
        Complex array of shape [n_states, n_actions, K].
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P
    K = len(omegas)

    reward_cf_sa = np.zeros((n_states, n_actions, K), dtype=complex)

    for s in range(n_states):
        for a in range(n_actions):
            cf_sa = np.zeros(K, dtype=complex)
            for prob, _, reward, _ in P[s][a]:
                cf_sa += prob * np.exp(1j * omegas * reward)
            reward_cf_sa[s, a] = cf_sa

    return reward_cf_sa


# ---------------------------------------------------------------------------
# CVI policy evaluation: V_pi(s, ω)
# ---------------------------------------------------------------------------

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
    interp_method: InterpolationMethod = "linear",
    interp_kwargs: dict = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Tabular Characteristic Value Iteration (CVI) for policy evaluation.

    This computes the fixed point V(s, ω) ≈ φ_{G}(s, ω) where G is the
    discounted return under policy π:

      G_π(s) = R_0 + γ R_1 + γ^2 R_2 + ...

    In the frequency domain, the Bellman equation becomes:

      V(s, ω) = φ_R(s, ω) * E[V(S', γ ω) | s, a=π(s)]

    We approximate this on a finite ω-grid, using various interpolation methods
    to evaluate V(s, γ ω) between grid points.
    """
    n_states = env_spec.n_states
    P: TransitionModel = env_spec.P

    if interp_kwargs is None:
        interp_kwargs = {}

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
            if interp_method == "linear":
                V_scaled[s] = interpolate_linear(scaled_omegas, omegas, V_prev[s])
            elif interp_method == "polar":
                V_scaled[s] = interpolate_polar(scaled_omegas, omegas, V_prev[s])
            elif interp_method == "pchip":
                V_scaled[s] = interpolate_pchip(scaled_omegas, omegas, V_prev[s])
            elif interp_method == "lanczos":
                V_scaled[s] = interpolate_lanczos(scaled_omegas, omegas, V_prev[s], **interp_kwargs)
            else:
                raise ValueError(f"Unknown interpolation method: {interp_method}")

        # 3b) Bellman update for each state
        max_delta = 0.0

        for s in range(n_states):
            a = int(policy[s])

            # Expectation over next states under P(s,a)
            exp_next = np.zeros(K, dtype=complex)
            for prob, next_state, _, done in P[s][a]:
                if done:
                    exp_next += prob * 1.0  # CF of zero future return
                else:
                    exp_next += prob * V_scaled[next_state]

            V[s] = reward_cf_table[s] * exp_next
            max_delta = max(max_delta, np.max(np.abs(V[s] - V_prev[s])))

        if max_delta < eps:
            break

    if return_omegas:
        return V, omegas

    return V, None


# ---------------------------------------------------------------------------
# CVI action evaluation: Q_cf(s,a,ω) given V(s,ω)
# ---------------------------------------------------------------------------

def cvi_action_evaluation_from_V(
    env_spec: TabularEnvSpec,
    V_cf: np.ndarray,
    omegas: np.ndarray,
    gamma: float = 0.9,
    interp_method: InterpolationMethod = "linear",
    interp_kwargs: dict = None
) -> np.ndarray:
    """
    Compute the CF action-values Q_cf(s,a,ω) from a given CF state-value V(s,ω).

    We use the CF form of the Bellman equation for Q:

      Q(s,a,ω) = φ_R(s,a,ω) * E[ V(S', γ ω) | s, a ]

    where:
      - φ_R(s,a,ω) is the one-step reward CF
      - V(s,ω) approximates the CF of the discounted return G starting at s

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment spec.
    V_cf : np.ndarray
        Complex array of shape [n_states, K] representing V(s, ω_k).
    omegas : np.ndarray
        Frequency grid, shape [K].
    gamma : float
        Discount factor.

    Returns
    -------
    Q_cf : np.ndarray
        Complex array of shape [n_states, n_actions, K], Q_cf[s,a,k] ≈ Q(s,a,ω_k).
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P
    K = len(omegas)

    if interp_kwargs is None:
        interp_kwargs = {}

    if V_cf.shape != (n_states, K):
        raise ValueError(
            f"V_cf shape {V_cf.shape} incompatible with (n_states, K)=({n_states}, {K})."
        )

    # 1) Reward CF φ_R(s,a,ω)
    reward_cf_sa = compute_reward_cf_table_sa(env_spec, omegas)  # [S,A,K]

    # 2) Interpolate V(s, γω) for all states
    scaled_omegas = gamma * omegas
    V_scaled = np.zeros_like(V_cf)

    for s in range(n_states):
        if interp_method == "linear":
            V_scaled[s] = interpolate_linear(scaled_omegas, omegas, V_cf[s])
        elif interp_method == "polar":
            V_scaled[s] = interpolate_polar(scaled_omegas, omegas, V_cf[s])
        elif interp_method == "pchip":
            V_scaled[s] = interpolate_pchip(scaled_omegas, omegas, V_cf[s])
        elif interp_method == "lanczos":
            V_scaled[s] = interpolate_lanczos(scaled_omegas, omegas, V_cf[s], **interp_kwargs)
        else:
            raise ValueError(f"Unknown interpolation method: {interp_method}")

    # 3) Build Q_cf(s,a,ω) via Bellman equation in CF-domain
    Q_cf = np.zeros((n_states, n_actions, K), dtype=complex)

    for s in range(n_states):
        for a in range(n_actions):
            exp_next = np.zeros(K, dtype=complex)
            for prob, next_state, _, done in P[s][a]:
                if done:
                    exp_next += prob * 1.0
                else:
                    exp_next += prob * V_scaled[next_state]

            Q_cf[s, a] = reward_cf_sa[s, a] * exp_next

    return Q_cf


# ---------------------------------------------------------------------------
# Collapse Q_cf(s,a,ω) to scalar Q(s,a) via mean of return
# ---------------------------------------------------------------------------

def collapse_q_cf_to_scalar_mean(
    omegas: np.ndarray,
    Q_cf: np.ndarray,
    method: CollapseMethod = "ls",
    **kwargs,
) -> np.ndarray:
    """
    Collapse Q_cf(s,a,ω) into scalar Q(s,a), using a variety of methods.

    Parameters
    ----------
    omegas : np.ndarray
        Frequency grid, shape [K].
    Q_cf : np.ndarray
        Complex CF action-values, shape [n_states, n_actions, K].
    method : CollapseMethod
        Method to use for mean estimation: "ls", "fft", "gaussian", "savgol".
    **kwargs : 
        Additional arguments passed to the estimator (e.g. m, max_w, window_length).

    Returns
    -------
    Q_mean : np.ndarray
        Real-valued Q-table of shape [n_states, n_actions].
    """
    n_states, n_actions, K = Q_cf.shape
    if len(omegas) != K:
        raise ValueError(
            f"omegas length {len(omegas)} does not match Q_cf last dim {K}."
        )

    Q_mean = np.zeros((n_states, n_actions), dtype=float)

    for s in range(n_states):
        for a in range(n_actions):
            phi_sa = Q_cf[s, a]
            
            if method == "ls":
                val = estimate_mean_ls(omegas, phi_sa, **kwargs)
            elif method == "fft":
                val = estimate_mean_fft(omegas, phi_sa)
            elif method == "gaussian":
                val = estimate_mean_gaussian(omegas, phi_sa, **kwargs)
            elif method == "savgol":
                val = estimate_mean_savgol(omegas, phi_sa, **kwargs)
            else:
                raise ValueError(f"Unknown collapse method: {method}")
                
            Q_mean[s, a] = val

    return Q_mean
