# cvi_rl/algorithms/tabular_cvi.py

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

import time

from cvi_rl.algorithms.mc import evaluate_policy_monte_carlo 
from cvi_rl.algorithms.utils import sample_initial_states

from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
from cvi_rl.cf.grids import make_omega_grid, GridStrategy
from cvi_rl.cf.processing import (
    CollapseMethod,
    InterpolationMethod,
    interpolate_cf,
    estimate_mean_from_cf,
)
from tqdm import tqdm

import matplotlib.pyplot as plt
try:
    import wandb
except ImportError:
    wandb = None


# ---------------------------------------------------------------------------
# Reward CFs
# ---------------------------------------------------------------------------

def compute_immediate_reward_cf_state_frequency(
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Compute reward characteristic functions per state under a fixed policy π.

    For each state s, we consider the one-step reward distribution R(s,π(s)):
      CF_R(s, ω) = E[exp(i ω R) | s, a=π(s)], see equation (3) CVI paper.

    For tabular MDPs with transition model P[s][a] = (prob, next_state, reward, done),
    this becomes:

      CF_R(s, ω) = sum_{(prob, *, r, *)} prob * exp(i ω r)

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
        is CF_R(s, ω_k).
    """
    
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P
    K = len(omegas)

    if policy.shape[0] != n_states:
        raise ValueError(
            f"Policy length {policy.shape[0]} does not match env states {n_states}."
        )

    immediate_reward_cf_table = np.zeros((n_states, K), dtype=complex)

    for s in range(n_states):
        a = int(policy[s])
        if not (0 <= a < n_actions):
            raise ValueError(f"Policy action {a} out of bounds for state {s}.")
        immediate_reward_cf_s = np.zeros(K, dtype=complex)
        for prob, _, reward, _ in P[s][a]:
            immediate_reward_cf_s += prob * np.exp(1j * omegas * reward)

        immediate_reward_cf_table[s] = immediate_reward_cf_s
    return immediate_reward_cf_table # [n_states, K]


def compute_immediate_reward_cf_state_action_frequency(
    env_spec: TabularEnvSpec,
    omegas: np.ndarray,
) -> np.ndarray:
    """
    Compute reward characteristic functions for every state-action pair:

      CF_R(s,a,ω) = E[exp(i ω R) | s, a]

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


#! I don't know if we shoud keep the default values... (I'd rather know I missed something)
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

    This computes the fixed point V(s, ω) ≈ CF_{G}(s, ω) where G is the
    discounted return under policy π:

      G_π(s) = R_0 + gamma R_1 + gamma^2 R_2 + ...

    In the frequency domain, the Bellman equation becomes:

      V(s, ω) = CF_R(s, ω) * E[V(S', gamma ω) | s, a=π(s)]

    We approximate this on a finite ω-grid, using various interpolation methods
    to evaluate V(s, gamma ω) between grid points.
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

    # 2) Precompute reward CF_R(s, ω)
    reward_cf_table = compute_immediate_reward_cf_state_frequency(env_spec, policy, omegas)

    # 3) Initialize V(s, ω) = 1 (CF of zero-return RV)
    V = np.ones((n_states, K), dtype=complex)

    for _ in range(max_iters):
        V_prev = V.copy()

        # 3a) Compute V(s, gamma * ω) via interpolation necessary for L2 loss bootstrapping
        V_cf_gamma = np.zeros_like(V_prev)
        
        for s in range(n_states):
            V_cf_gamma[s] = interpolate_cf(scaled_omegas, omegas, V_prev[s], interp_method, **interp_kwargs)
        
        # 3b) Bellman update for each state
        max_delta = 0.0

        for s in range(n_states):
            a = int(policy[s])

            # Expectation over next states under P(s,a)
            expectation_over_next_state = np.zeros(K, dtype=complex)
            for prob, next_state, _, done in P[s][a]:
                if done:
                    expectation_over_next_state += prob * 1.0  #! Multiplying by 1 to make CF of zero future return
                else:
                    expectation_over_next_state += prob * V_cf_gamma[next_state]

            # Time domain: V(s) = R + gamma * E[V(S')]
            # Frequency domain: V(s,ω) = CF_R(s,ω) * E[V(S', gamma ω)]
            V[s] = reward_cf_table[s] * expectation_over_next_state
            max_delta = max(max_delta, np.max(np.abs(V[s] - V_prev[s])))

        if max_delta < float(eps):
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
    gamma: float,
    interp_method: InterpolationMethod,
    interp_kwargs: dict
) -> np.ndarray:
    """
    Compute the CF action-values Q_cf(s,a,ω) from a given CF state-value V(s,ω).

    We use the CF form of the Bellman equation for Q:

      Q(s,a,ω) = CF_R(s,a,ω) * E[ V(S', gamma ω) | s, a ]

    where:
      - CF_R(s,a,ω) is the one-step reward CF
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

    #! 1) Reward CF CF_R(s,a,ω)
    #!reward_cf_sa = compute_immediate_reward_cf_state_action_frequency(env_spec, omegas)  # [n_states, n_actions, K]

    # 2) Compute V(s, gamma * ω) via interpolation necessary for L2 loss bootstrapping
    scaled_omegas = gamma * omegas
    V_cf_gamma = np.zeros_like(V_cf)

    for s in range(n_states):
        V_cf_gamma[s] = interpolate_cf(scaled_omegas, omegas, V_cf[s], interp_method, **interp_kwargs)

    # 3) Build Q_cf(s,a,ω) via Bellman equation in CF-domain
    Q_cf = np.zeros((n_states, n_actions, K), dtype=complex)
    
    for s in range(n_states):
        for a in range(n_actions):
            q_sa = np.zeros(K, dtype=complex)
            
            for prob, next_state, reward, done in P[s][a]:
                term = np.exp(1j * omegas * reward)
                if done:
                    pass
                else:
                    term *= V_cf_gamma[next_state]
                
                q_sa += prob * term

            Q_cf[s, a] = q_sa

    return Q_cf

    # for s in range(n_states):
    #     for a in range(n_actions):
    #         exp_next = np.zeros(K, dtype=complex)
    #         for prob, next_state, _, done in P[s][a]:
    #             if done:
    #                 exp_next += prob * 1.0 #! Multiplying by 1 to make CF of zero future return
    #             else:
    #                 exp_next += prob * V_cf_gamma[next_state]

    #         # Time domain: Q(s,a) = R + gamma * E[Q(s', a')]
    #         # Frequency domain: Q(s,a,ω) = CF_R(s,a,ω) * E[Q(s', a' gamma ω)]
    #         Q_cf[s, a] = reward_cf_sa[s, a] * exp_next

    # return Q_cf


# ---------------------------------------------------------------------------
# Collapse Q_cf(s,a,ω) to scalar Q(s,a) via mean of return
# ---------------------------------------------------------------------------

def collapse_q_cf_to_scalar_mean(
    omegas: np.ndarray,
    Q_cf: np.ndarray,
    method: CollapseMethod,
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
            Q_cf_sa = Q_cf[s, a]
            val = estimate_mean_from_cf(omegas, Q_cf_sa, method, **kwargs) 
            Q_mean[s, a] = val

    return Q_mean

def run_cvi(env_spec: TabularEnvSpec, env, config: dict, logger=None):
    """
    Run CVI as a value iteration loop: evaluate V_cf → collapse to Q → greedy policy → repeat.
    Includes MC evaluation for metrics, like in PI.
    """
        
    gamma = config['gamma']
    grid_strategy = config['grid_strategy']
    W = config['W']
    K = config['K']
    interp_method = config['interp_method']
    collapse_method = config['collapse_method']
    eval_termination = config['eval_termination']  # For CF convergence
    max_iters = config['max_iters']  # Max VI iterations
    eval_episodes = config['eval_episodes']  # For final MC eval
    max_steps = config['max_steps']  # Max steps per episode
    seed = config.get('seed', None)
    
    print(f"Running CVI Value Iteration: {grid_strategy} grid, K={K}, W={W}")
    
    n_states = env_spec.n_states
    start_time = time.time()
    
    policy = np.zeros(n_states, dtype=int)
        
    v_history = []
    
    for iter_num in tqdm(range(max_iters), desc="CVI Value Iteration"):
        policy_prev = policy.copy()
        
        # 1) Evaluate V_cf for current policy
        V_cf, omegas = cvi_policy_evaluation(
            env_spec, policy, gamma=gamma,
            grid_strategy=grid_strategy, W=W, K=K,
            interp_method=interp_method, eps=eval_termination
        )
        
        # 2) Compute Q_cf from V_cf
        Q_cf = cvi_action_evaluation_from_V(
            env_spec, V_cf, omegas, gamma=gamma, interp_method=interp_method, interp_kwargs=None
        )
        
        # 3) Collapse Q_cf to scalar Q
        #! This function can be an error source
        Q_scalar = collapse_q_cf_to_scalar_mean(omegas, Q_cf, method=collapse_method)
        
        # 4) Greedy policy improvement
        policy = np.argmax(Q_scalar, axis=1)
        mean_v = np.mean(np.max(Q_scalar, axis=1))  # Mean of state values (max Q per state)
        v_history.append(mean_v)

        if logger:
            log_dict = {'mean_v_value': float(mean_v)}
            logger(log_dict, step=iter_num + 1)
            
        # Check convergence (policy stable)
        if np.array_equal(policy, policy_prev):
            print(f"Converged at iteration {iter_num + 1}")
            break
    
    elapsed_time = time.time() - start_time
    
    states_to_evaluate = sample_initial_states(env, eval_episodes)
    cvi_expected_from_reset = float(np.mean(np.max(Q_scalar[states_to_evaluate], axis=1)))
    
    if eval_episodes > 0 and env is not None:
        avg_return, var_return, success_rate, returns, avg_steps, _ = evaluate_policy_monte_carlo(
            env, env_spec, policy, n_episodes=eval_episodes, gamma=gamma, max_steps=max_steps, 
            seed=seed, states_to_evaluate=states_to_evaluate
        )
        mc_metrics = {
            'mc_avg_return': float(avg_return),
            'mc_success_rate': float(success_rate),
            # 'mc_var_return': float(var_return),
        }
        
        if logger:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot MC Histogram
                ax.hist(returns, bins=50, density=True, alpha=0.5, color='gray', label='Monte Carlo (Ground Truth)')
                
                # Plot CVI PDF for State 0 (assuming it's the start state)
                target_state = 0
                xs, pdf = get_pdf_from_cf(omegas, V_cf[target_state])
                ax.plot(xs, pdf, color='blue', linewidth=2, label=f'CVI Estimate (State {target_state})')
                
                ax.set_xlim(-2, 2)
                ax.set_title(f"Return Distribution Comparison (State {target_state})")
                ax.set_xlabel("Return")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Log to wandb
                if wandb and wandb.run:
                    logger({'distribution_plot': wandb.Image(fig)})
                
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Could not generate distribution plot: {e}")

    else:
        mc_metrics = {}
    
    metrics = {
        'training_time': elapsed_time,
        'converged_iterations': len(v_history),
        'expected_initial_state_value': cvi_expected_from_reset,
        "final_mean_v_value": float(np.mean(v_history[-1])),
        **mc_metrics
    }
    
    if logger:
        logger(metrics) 
    
    print(f"\nConverged in {metrics['converged_iterations']} iterations")
    print(f"Training time: {elapsed_time:.2f}s")
    
    return {
        'policy': policy,
        'V_cf': V_cf,
        'Q_scalar': Q_scalar,
        'omegas': omegas,
        'metrics': metrics
    }

def get_pdf_from_cf(omegas: np.ndarray, cf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert a Characteristic Function to get the PDF (Probability Density Function).
    """
    d_omega = omegas[1] - omegas[0]
    K = len(omegas)
    
    # 1. Shift CF for IFFT (assuming omegas are centered around 0)
    cf_shifted = np.fft.ifftshift(cf)
    
    # 2. Inverse FFT to get PDF
    # We multiply by K/range to normalize, but for visualization relative shape matters most
    pdf_complex = np.fft.ifft(cf_shifted)
    pdf_real = np.real(pdf_complex)
    
    # 3. Shift PDF to center the domain
    pdf_real = np.fft.fftshift(pdf_real)
    
    # 4. Construct x-axis (Return values)
    # The span of the spatial domain is 2*pi / d_omega
    L = 2 * np.pi / d_omega
    xs = np.linspace(-L/2, L/2, K, endpoint=False)
    
    # 5. Normalize to sum to 1 (to treat as probabilities/counts for histogram)
    # Clip negatives that might appear due to numerical noise
    pdf_real = np.maximum(pdf_real, 0)
    pdf_real = pdf_real / (np.sum(pdf_real) + 1e-9)
    
    return xs, pdf_real