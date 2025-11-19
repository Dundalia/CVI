# cvi_rl/algorithms/tabular_q.py

from __future__ import annotations

from typing import Tuple, List, Optional
import numpy as np
import gymnasium as gym
from cvi_rl.envs.base import TabularEnvSpec


def q_learning(
    env: gym.Env,
    env_spec: TabularEnvSpec,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    epsilon_decay: Optional[float] = None,
    max_steps: int = 10_000,
    track_q_every: Optional[int] = 10,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[np.ndarray]]:
    """
    Tabular Q-learning.

    Parameters
    ----------
    env : gym.Env
        Environment instance (e.g., Taxi-v3, FrozenLake, ...).
    env_spec : TabularEnvSpec
        Tabular environment spec (for n_states, n_actions).
    num_episodes : int
        Number of training episodes.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial ε for ε-greedy exploration.
    epsilon_decay : float, optional
        Per-episode multiplicative decay factor for ε (e.g., 0.999).
    max_steps : int
        Max steps per episode (to avoid infinite loops).
    track_q_every : int, optional
        If not None, store a snapshot of Q every `track_q_every` episodes.

    Returns
    -------
    policy : np.ndarray
        Greedy policy derived from final Q, shape [n_states].
    Q : np.ndarray
        Learned action-value function Q(s,a), shape [n_states, n_actions].
    episode_returns : list[float]
        Total (undiscounted) return per episode.
    intermediate_qs : list[np.ndarray]
        Snapshots of Q at intermediate training stages.
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions

    Q = np.zeros((n_states, n_actions), dtype=float)
    episode_returns: List[float] = []
    intermediate_qs: List[np.ndarray] = []

    eps = float(epsilon)

    for episode in range(num_episodes):
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out

        done = False
        episode_return = 0.0
        steps = 0

        # Decay epsilon if specified
        if epsilon_decay is not None:
            eps *= epsilon_decay

        while not done and steps < max_steps:
            # ε-greedy action selection
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            step_out = env.step(action)

            # Gymnasium vs gym handling
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            # Q-learning update
            best_next_action = int(np.argmax(Q[next_state]))
            td_target = reward + gamma * Q[next_state, best_next_action] * (1 - done)
            td_delta = td_target - Q[state, action]
            Q[state, action] += alpha * td_delta

            state = next_state
            episode_return += reward
            steps += 1

        episode_returns.append(float(episode_return))

        # Store intermediate Q-values every N episodes
        if track_q_every is not None and (episode + 1) % track_q_every == 0:
            intermediate_qs.append(Q.copy())

    # Derive greedy policy from final Q-table
    policy = np.argmax(Q, axis=1)

    return policy, Q, episode_returns, intermediate_qs