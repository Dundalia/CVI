from __future__ import annotations

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
from cvi_rl.algorithms.mc import evaluate_policy_monte_carlo
from cvi_rl.algorithms.utils import sample_initial_states
from cvi_rl.algorithms.tabular_vi import value_iteration


def _make_atoms(v_min: float, v_max: float, n_atoms: int) -> Tuple[np.ndarray, float]:
    if n_atoms < 2:
        raise ValueError("n_atoms must be >= 2")
    if v_max <= v_min:
        raise ValueError("v_max must be > v_min")
    atoms = np.linspace(v_min, v_max, n_atoms, dtype=float)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    return atoms, float(delta_z)


def _project_distribution_to_atoms(
    tz: np.ndarray,         # [N] transformed atom locations
    p: np.ndarray,          # [N] probability mass for those atoms
    atoms: np.ndarray,      # [N]
    v_min: float,
    v_max: float,
    delta_z: float,
) -> np.ndarray:
    """
    Standard C51 projection of (tz, p) onto fixed support atoms.

    Returns projected probabilities m of shape [N].
    """
    N = atoms.shape[0]
    if tz.shape != (N,) or p.shape != (N,):
        raise ValueError(f"tz and p must be shape ({N},), got {tz.shape}, {p.shape}")

    # Clip onto support range
    tz = np.clip(tz, v_min, v_max)

    b = (tz - v_min) / delta_z
    l = np.floor(b).astype(np.int64)
    u = np.ceil(b).astype(np.int64)
    l = np.clip(l, 0, N - 1)
    u = np.clip(u, 0, N - 1)

    m = np.zeros(N, dtype=float)

    eq = (l == u)
    if np.any(eq):
        np.add.at(m, l[eq], p[eq])

    neq = ~eq
    if np.any(neq):
        # linear interpolation
        np.add.at(m, l[neq], p[neq] * (u[neq] - b[neq]))
        np.add.at(m, u[neq], p[neq] * (b[neq] - l[neq]))

    # Numerical cleanup (projection should preserve mass but clipping can cause tiny drift)
    s = float(np.sum(m))
    if s > 0:
        m /= s
    return m


def _project_scalar_to_atoms(
    x: float,
    atoms: np.ndarray,
    v_min: float,
    v_max: float,
    delta_z: float,
) -> np.ndarray:
    """
    Project a deterministic return (delta at x) onto the categorical atoms.
    """
    N = atoms.shape[0]
    x = float(np.clip(x, v_min, v_max))

    b = (x - v_min) / delta_z
    l = int(np.floor(b))
    u = int(np.ceil(b))
    l = int(np.clip(l, 0, N - 1))
    u = int(np.clip(u, 0, N - 1))

    m = np.zeros(N, dtype=float)
    if l == u:
        m[l] = 1.0
        return m

    m[l] = (u - b)
    m[u] = (b - l)

    s = float(np.sum(m))
    if s > 0:
        m /= s
    return m


def categorical_distributional_value_iteration(
    env_spec: TabularEnvSpec,
    gamma: float,
    v_min: float,
    v_max: float,
    n_atoms: int = 51,
    max_iters: int = 200,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Tabular C51-style Distributional Value Iteration (control).

    Stores Z[s,a] as a categorical distribution over fixed atoms.
    Improves policy using argmax of expected return (mean of Z).

    Returns
    -------
    policy : [n_states] int
    Z      : [n_states, n_actions, n_atoms] float
    atoms  : [n_atoms] float
    value_history : list[np.ndarray] 
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P

    atoms, delta_z = _make_atoms(v_min, v_max, n_atoms)
    value_history: List[np.ndarray] = []

    # Initialize Z to a delta at 0 (projected onto atoms), for all (s,a)
    init_dist = _project_scalar_to_atoms(0.0, atoms, v_min, v_max, delta_z)
    Z = np.tile(init_dist, (n_states, n_actions, 1)).astype(float)

    policy = np.zeros(n_states, dtype=int)

    for _ in range(max_iters):
        Z_prev = Z.copy()

        # Greedy policy under current distribution means
        Q_mean = np.tensordot(Z_prev, atoms, axes=([2], [0]))  # [S, A]
        policy = np.argmax(Q_mean, axis=1).astype(int)

        V_mean = np.max(Q_mean, axis=1) # [S] max over actions
        value_history.append(V_mean.copy())

        max_change = 0.0

        # Bellman optimality backup in distribution space
        for s in range(n_states):
            for a in range(n_actions):
                m_sa = np.zeros(n_atoms, dtype=float)

                for prob, next_state, reward, done in P[s][a]:
                    prob = float(prob)
                    reward = float(reward)
                    done = bool(done)

                    if done:
                        # Terminal: return is deterministic "reward"
                        m_sa += prob * _project_scalar_to_atoms(
                            reward, atoms, v_min, v_max, delta_z
                        )
                    else:
                        # Choose greedy action at next state (under mean)
                        a_star = int(policy[next_state])
                        p_next = Z_prev[next_state, a_star]  # [N]

                        tz = reward + gamma * atoms  # transformed atom locations
                        m_sa += prob * _project_distribution_to_atoms(
                            tz=tz,
                            p=p_next,
                            atoms=atoms,
                            v_min=v_min,
                            v_max=v_max,
                            delta_z=delta_z,
                        )

                # Normalize (should already be close to 1, but keep it safe)
                s_mass = float(np.sum(m_sa))
                if s_mass > 0:
                    m_sa /= s_mass

                Z[s, a] = m_sa
                max_change = max(max_change, float(np.sum(np.abs(Z[s, a] - Z_prev[s, a]))))

        if max_change < float(eps):
            break

    # Final greedy policy
    Q_mean = np.tensordot(Z, atoms, axes=([2], [0]))  # [S, A]
    policy = np.argmax(Q_mean, axis=1).astype(int)
    return policy, Z, atoms, value_history


def run_c51(env_spec: TabularEnvSpec, env, config: Dict[str, Any], logger=None):
    """
    Wrapper consistent with your other runners.

    Expected config keys (suggested):
      gamma: float
      v_min: float
      v_max: float
      n_atoms: int (e.g., 51)
      max_iters: int
      eval_termination: float  (used as eps)
      eval_episodes: int
      max_steps: int
      seed: Optional[int]
    """
    gamma = float(config["gamma"])
    v_min = float(config["v_min"])
    v_max = float(config["v_max"])
    n_atoms = int(config["n_atoms"])
    max_iters = int(config["max_iters"])
    eps = float(config["eval_termination"])

    eval_episodes = int(config["eval_episodes"])
    max_steps = int(config["max_steps"])
    seed = config.get("seed", None)

    print("\n" + "=" * 60)
    print("Running Tabular C51 (Categorical Distributional VI)")
    print("=" * 60)
    print(f"  Gamma: {gamma}")
    print(f"  Support: [{v_min}, {v_max}] with n_atoms={n_atoms}")
    print(f"  Max iters: {max_iters}, eps: {eps}")

    start = time.time()
    
    # Compute true optimal value function for error calculation
    _, optimal_V, _, _ = value_iteration(
        env_spec,
        gamma,
        iterations=10000,  # High max iters
        termination=1e-12,  # Tight convergence
        track_history=False,
    )
    policy, Z, atoms, value_history = categorical_distributional_value_iteration(
        env_spec=env_spec,
        gamma=gamma,
        v_min=v_min,
        v_max=v_max,
        n_atoms=n_atoms,
        max_iters=max_iters,
        eps=eps
    )
    elapsed = time.time() - start

    # Scalar values from distribution means
    Q_mean = np.tensordot(Z, atoms, axes=([2], [0]))  # [S, A]
    V_mean = np.max(Q_mean, axis=1)                  # [S]
    
    
    states_to_evaluate = sample_initial_states(env, eval_episodes)
    expected_v_from_initial_states = float(np.mean(V_mean[states_to_evaluate]))

    metrics = {
        "training_time": float(elapsed),
        "converged_iterations": len(value_history),
        "expected_initial_state_value": expected_v_from_initial_states,
        "final_mean_v_value": float(np.mean(V_mean))
    }

    # Log the value function history
    if logger and value_history is not None:
        for i in range(1, len(value_history)):
            td_error = np.max(np.abs(value_history[i] - optimal_V))
            logger({'mean_v_value': float(np.mean(value_history[i])), 'td_error': td_error}, step=i)

    if eval_episodes > 0:
        avg_return, var_return, success_rate, returns, avg_steps, var_steps = evaluate_policy_monte_carlo(
            env,
            env_spec,
            policy,
            n_episodes=eval_episodes,
            states_to_evaluate=states_to_evaluate,
            gamma=gamma,
            max_steps=max_steps,
            seed=seed,
        )
        metrics.update(
            {
                "mc_avg_return": float(avg_return),
                "mc_success_rate": float(success_rate),
            }
        )

        if logger:
            plot_cdf_comparison(logger, returns, atoms, Z, policy, wandb)

    if logger:
        logger(metrics)
        
    return {
        "policy": policy,
        "Z": Z,             # [S, A, N] categorical distributions
        "atoms": atoms,     # [N]
        "Q_mean": Q_mean,   # [S, A]
        "V_mean": V_mean,   # [S]
        "metrics": metrics,
    }


def plot_cdf_comparison(logger, returns, atoms, Z, policy, wandb_module, save_path="figures/cdf_c51.png"):
    try:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 1. Plot MC Empirical CDF
        sorted_returns = np.sort(returns)
        yvals = np.arange(len(sorted_returns)) / float(len(sorted_returns) - 1)
        ax.step(sorted_returns, yvals, color='gray', linestyle='--', label='Monte Carlo (Ground Truth)', where='post')
        
        # 2. Plot C51 CDF for State 0
        target_state = 0
        action = policy[target_state]
        probs = Z[target_state, action]
        
        # Compute CDF
        cdf = np.cumsum(probs)
        
        # Plot as a step function
        ax.step(atoms, cdf, color='red', linewidth=2, label=f'C51 (State {target_state})', where='post')
        
        ax.set_xlim(0, 1.0) # FrozenLake returns are usually in [0, 1]
        ax.set_title("C51", fontweight='bold', fontsize=14)
        ax.set_xlabel("Return", fontweight='bold', fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontweight='bold', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save locally
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved CDF plot to {save_path}")
        
        if wandb_module and wandb_module.run:
            logger({'distribution_plot': wandb_module.Image(fig)})
        
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate distribution plot: {e}")
