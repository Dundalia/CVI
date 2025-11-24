# cvi_rl/algorithms/tabular_quantile.py

from __future__ import annotations

import numpy as np
import time
from tqdm import tqdm
from cvi_rl.envs.base import TabularEnvSpec
from cvi_rl.algorithms.mc import evaluate_policy_monte_carlo

try:
    import wandb
except ImportError:
    wandb = None


def compute_quantiles_from_mixture(
    values: np.ndarray,
    weights: np.ndarray,
    n_quantiles: int
) -> np.ndarray:
    """
    Given a mixture distribution defined by (values, weights),
    compute the N quantiles at tau_i = (2i - 1) / (2N).
    
    Parameters
    ----------
    values : np.ndarray
        Array of values (locations of Diracs).
    weights : np.ndarray
        Array of probabilities associated with each value.
    n_quantiles : int
        Number of quantiles to return.
        
    Returns
    -------
    quantiles : np.ndarray
        Array of shape [n_quantiles].
    """
    # 1. Sort values and reorder weights
    sort_idx = np.argsort(values)
    sorted_values = values[sort_idx]
    sorted_weights = weights[sort_idx]
    
    # 2. Compute CDF
    cdf = np.cumsum(sorted_weights)
    
    # 3. Target cumulative probabilities for quantiles
    # tau_i = (i + 0.5) / N for i in 0..N-1? 
    # Standard Quantile Regression uses midpoints: (2i + 1) / 2N
    # i ranges 0 to N-1.
    # i=0: 1/2N. i=N-1: (2N-1)/2N.
    taus = (2 * np.arange(n_quantiles) + 1) / (2 * n_quantiles)
    
    # 4. Find values corresponding to taus (Inverse CDF)
    # We want the first value where CDF >= tau
    indices = np.searchsorted(cdf, taus)
    
    # Clip indices just in case (though taus < 1.0 and cdf[-1] should be 1.0)
    indices = np.clip(indices, 0, len(values) - 1)
    
    return sorted_values[indices]


def run_quantile_vi(env_spec: TabularEnvSpec, env, config: dict, logger=None):
    """
    Run Tabular Quantile Value Iteration.
    """
    print("\n" + "="*60)
    print("Running Quantile Value Iteration (QVI)")
    print("="*60)
    
    # Config
    gamma = config['gamma']
    max_iters = config['max_iters']
    eval_termination = config['eval_termination']
    n_quantiles = config['n_quantiles']
    
    # MC Eval
    eval_episodes = config.get('eval_episodes', 0)
    max_steps = config.get('max_steps', 100)
    initial_state = config.get('initial_state', None)
    init_policy = config.get('init_policy', None)
    seed = config.get('seed', None)
    
    print(f"  N Quantiles: {n_quantiles}")
    print(f"  Gamma: {gamma}")
    
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P = env_spec.P
    
    # Initialize Quantiles Z(s,a)
    # Shape: [n_states, n_actions, n_quantiles]
    # Initialize to zeros (or small random values)
    Z = np.zeros((n_states, n_actions, n_quantiles))
    
    start_time = time.time()
    
    if init_policy is not None:
        policy = np.array(init_policy, dtype=int, copy=True)
    else:
        policy = np.random.randint(0, env_spec.n_actions, size=n_states)
    
    v_history = []
    
    # For logging
    log_state = initial_state if initial_state is not None else 0
    
    for iter_num in tqdm(range(max_iters), desc="Quantile VI"):
        Z_new = np.zeros_like(Z)
        
        # 1. Compute greedy policy based on current mean Q-values
        # Mean of quantiles is the mean of the distribution
        Q_means = np.mean(Z, axis=2)
        policy = np.argmax(Q_means, axis=1)
        
        # Log progress
        mean_v = np.mean(np.max(Q_means, axis=1))
        v_history.append(mean_v)
        
        if logger and iter_num % 10 == 0:
            best_a = policy[log_state]
            dist_values = Z[log_state, best_a]
            
            # Log histogram of the particles/quantiles
            logger({
                'mean_v_value': float(mean_v),
                f'dist_state_{log_state}_action_{best_a}': wandb.Histogram(dist_values)
            }, step=iter_num + 1)
        elif logger:
            logger({'mean_v_value': float(mean_v)}, step=iter_num + 1)
            
        # 2. Bellman Update
        # For each (s, a), construct the target mixture distribution
        for s in range(n_states):
            for a in range(n_actions):
                
                # Collect all next particles
                all_next_values = []
                all_next_weights = []
                
                for prob, next_state, reward, done in P[s][a]:
                    if prob == 0: continue
                    
                    if done:
                        # Terminal: Deterministic reward
                        # We add N particles all equal to 'reward' with total weight 'prob'
                        # Or just 1 particle with weight 'prob' (since it's a Dirac)
                        all_next_values.append(reward)
                        all_next_weights.append(prob)
                    else:
                        # Bootstrap: Z(s', pi(s'))
                        # We take the quantiles from the next state's best action
                        next_a = policy[next_state]
                        next_quantiles = Z[next_state, next_a]
                        
                        # Shift by reward and gamma
                        shifted_values = reward + gamma * next_quantiles
                        
                        # Each quantile has weight 1/N * prob
                        # We can add N values, each with weight prob/N
                        all_next_values.extend(shifted_values)
                        all_next_weights.extend([prob / n_quantiles] * n_quantiles)
                
                # Convert to arrays
                all_next_values = np.array(all_next_values)
                all_next_weights = np.array(all_next_weights)
                
                # Compute new quantiles for Z(s,a)
                Z_new[s, a] = compute_quantiles_from_mixture(
                    all_next_values, all_next_weights, n_quantiles
                )
        
        # Check convergence on Mean Q-values
        Q_means_new = np.mean(Z_new, axis=2)
        diff = np.max(np.abs(Q_means_new - Q_means))
        
        Z = Z_new
        
        if diff < float(eval_termination):
            print(f"Converged at iteration {iter_num + 1}")
            break
            
    elapsed_time = time.time() - start_time
    
    # Final Policy
    Q_means = np.mean(Z, axis=2)
    policy = np.argmax(Q_means, axis=1)
    
    # MC Eval
    mc_metrics = {}
    if eval_episodes > 0 and env is not None:
        avg_return, var_return, success_rate, _, avg_steps, _ = evaluate_policy_monte_carlo(
            env, env_spec, policy, n_episodes=eval_episodes, gamma=gamma, max_steps=max_steps, initial_state=initial_state, seed=seed
        )
        mc_metrics = {
            'mc_avg_return': float(avg_return),
            'mc_success_rate': float(success_rate),
        }
        
    metrics = {
        'training_time': elapsed_time,
        'converged_iterations': len(v_history),
        'final_mean_value': float(np.mean(np.max(Q_means, axis=1))),
        **mc_metrics
    }
    
    if logger:
        logger(metrics)
        
    return {
        'policy': policy,
        'Z': Z,
        'Q_values': Q_means,
        'metrics': metrics
    }
