# cvi_rl/algorithms/mc.py

from __future__ import annotations

from typing import Tuple, List, Optional
import numpy as np
import gymnasium as gym
from cvi_rl.envs.base import TabularEnvSpec
from tqdm import tqdm


def run_episode_with_policy(
    env: gym.Env,
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    initial_state: Optional[int],
    gamma: float,
    max_steps: int,
) -> Tuple[float, bool, int]:
    """
    Run a single episode following a given tabular policy.

    Parameters
    ----------
    env : gym.Env
        The underlying environment instance (e.g., Taxi-v3).
    env_spec : TabularEnvSpec
        Tabular description of the environment.
    policy : np.ndarray
        Array of shape [n_states] mapping state -> action.
    initial_state : int, optional
        If provided, the environment state is forced to this value.
        Otherwise, env_spec.initial_state is used if available, else the env's default reset.
    gamma : float
        Discount factor for computing the return.
    max_steps : int
        Maximum number of steps before truncating the episode.

    Returns
    -------
    discounted_return : float
        Discounted cumulative reward.
    success : bool
        Heuristic success flag: True if episode terminated (not truncated)
        and final reward > 0 (for Taxi this corresponds to a successful dropoff).
    steps : int
        Number of steps taken in the episode.
    """
    # Reset env
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        state, _ = reset_out
    else:
        state = reset_out

    # Choose initial state
    if initial_state is None:
        initial_state = env_spec.initial_state if env_spec.initial_state is not None else state

    # Force underlying state if supported (Taxi/FrozenLake style)
    if hasattr(env.unwrapped, "s"):
        env.unwrapped.s = initial_state
        state = initial_state
    else:
        # fall back to whatever env.reset gave us
        state = state

    rewards: List[float] = []
    done = False
    steps = 0
    last_reward = 0.0

    while not done and steps < max_steps:
        action = int(policy[state])
        step_out = env.step(action)

        # Gymnasium: obs, reward, terminated, truncated, info
        # Classic gym: obs, reward, done, info
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_out

        rewards.append(reward)
        last_reward = reward
        state = next_state
        steps += 1

    discounted_return = 0.0
    for t, r in enumerate(rewards):
        discounted_return += (gamma ** t) * r

    # Generic "success": terminated with a positive final reward
    success = (steps > 0) and done and (last_reward > 0.0)

    return float(discounted_return), bool(success), int(steps)


def evaluate_policy_monte_carlo(
    env: gym.Env,
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    initial_state: Optional[int],
    n_episodes: int,
    gamma: float,
    max_steps: int,
    seed: Optional[int] = None,
):
    """
    Monte Carlo evaluation of a policy on a tabular environment.

    Parameters
    ----------
    env : gym.Env
        Environment instance.
    env_spec : TabularEnvSpec
        Tabular environment spec (for initial_state, etc.).
    policy : np.ndarray
        Policy mapping state -> action.
    initial_state : int, optional
        Optional fixed starting state; falls back to env_spec.initial_state.
    n_episodes : int
        Number of Monte Carlo episodes.
    gamma : float
        Discount factor.
    max_steps : int
        Max steps per episode.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    episodes_return: [float]
    avg_return : float
    var_return : float
    success_rate : float
    returns : list[float]
    avg_steps : float
    var_steps : float
    """
    returns: List[float] = []
    successes: List[bool] = []
    steps_list: List[int] = []
    
    # Seed the environment if provided
    if seed is not None:
        env.reset(seed=seed)

    for _ in tqdm(range(n_episodes), desc="MC Evaluation"):
        disc_return, success, steps = run_episode_with_policy(
            env,
            env_spec,
            policy,
            initial_state=initial_state,
            gamma=gamma,
            max_steps=max_steps,
        )
        returns.append(disc_return)
        successes.append(success)
        steps_list.append(steps)

    episodes_return = np.array(returns, dtype=float)
    steps_arr = np.array(steps_list, dtype=float)
    successes_arr = np.array(successes, dtype=float)

    avg_return = float(np.mean(episodes_return))
    var_return = float(np.var(episodes_return))
    success_rate = float(np.mean(successes_arr))
    avg_steps = float(np.mean(steps_arr))
    var_steps = float(np.var(steps_arr))

    return avg_return, var_return, success_rate, returns, avg_steps, var_steps


def run_monte_carlo(env_spec: TabularEnvSpec, env: gym.Env, config: dict, logger=None):
    """
    Train (evaluate) a random policy using Monte Carlo sampling.
    
    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification.
    env : gym.Env
        Gymnasium environment instance.
    config : dict
        Training configuration with keys:
        - gamma: float (discount factor)
        - n_episodes: int (number of episodes)
        - max_steps: int (max steps per episode)
    logger : callable, optional
        Function to log metrics, signature: logger(metrics_dict, step=None)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - policy: Random policy used
        - metrics: Performance metrics
        - returns: List of episode returns
    """
    import time
    try:
        import wandb
    except ImportError:
        wandb = None
    
    print("\n" + "="*60)
    print("Running Monte Carlo Evaluation")
    print("="*60)
    
    # Random policy
    policy = np.random.randint(0, env_spec.n_actions, size=env_spec.n_states)
    
    # Extract config
    gamma = config.get('gamma', 0.99)
    n_episodes = config.get('n_episodes', 1000)
    max_steps = config.get('max_steps', 200)
    
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Gamma: {gamma}")
    
    start_time = time.time()
    
    avg_return, var_return, success_rate, returns, avg_steps, var_steps = evaluate_policy_monte_carlo(
        env,
        env_spec,
        policy,
        initial_state=None,
        n_episodes=n_episodes,
        gamma=gamma,
        max_steps=max_steps
    )
    
    elapsed_time = time.time() - start_time
    
    metrics = {
        'algorithm': 'monte_carlo',
        'n_episodes': n_episodes,
        'training_time': elapsed_time,
        'avg_return': avg_return,
        'var_return': var_return,
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'var_steps': var_steps,
    }
    
    if logger:
        # Log histogram of returns
        if wandb is not None:
            metrics['returns_hist'] = wandb.Histogram(returns)
        logger(metrics)
    
    print(f"\nAvg Return: {avg_return:.3f}")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Training time: {elapsed_time:.2f}s")
    
    return {
        'policy': policy,
        'metrics': metrics,
        'returns': returns
    }