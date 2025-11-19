# cvi_rl/algorithms/tabular_vi.py

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from cvi_rl.envs.base import TabularEnvSpec, TransitionModel


def value_iteration(
    env_spec: TabularEnvSpec,
    gamma: float,
    initial_values: Optional[np.ndarray] = None,
    iterations: int = 100,
    termination: float = 1e-4,
    track_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    Classical tabular Value Iteration for a TabularEnvSpec.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification (n_states, n_actions, P, ...).
    gamma : float
        Discount factor in [0,1).
    initial_values : np.ndarray, optional
        Initial value function, shape [n_states]. If None, initialized to zeros.
    iterations : int
        Maximum number of iterations.
    termination : float
        Stop if max |V_{k+1} - V_k| < termination.
    track_history : bool
        If True, returns value_history and max_change_history.

    Returns
    -------
    policy : np.ndarray
        Greedy policy w.r.t. final value function, shape [n_states].
    values : np.ndarray
        Final value function, shape [n_states].
    value_history : list[np.ndarray] or None
        If track_history=True, list of value snapshots.
    max_change_history : list[float] or None
        If track_history=True, list of max updates per iteration.
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P

    if initial_values is None:
        values = np.zeros(n_states, dtype=float)
    else:
        values = np.array(initial_values, dtype=float, copy=True)

    policy = np.zeros(n_states, dtype=int)

    value_history: Optional[List[np.ndarray]] = [] if track_history else None
    max_change_history: Optional[List[float]] = [] if track_history else None

    for k in range(iterations):
        if track_history and value_history is not None:
            value_history.append(values.copy())

        max_update = 0.0

        # In-place asynchronous updates
        for s in range(n_states):
            old_value = values[s]
            q_values = []

            for a in range(n_actions):
                q_sa = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    # if done, no future value
                    q_sa += prob * (reward + gamma * values[next_state] * (1 - done))
                q_values.append(q_sa)

            values[s] = max(q_values)
            max_update = max(max_update, abs(values[s] - old_value))

        if track_history and max_change_history is not None:
            max_change_history.append(max_update)

        if max_update < termination:
            break

    # Extract greedy policy from final values
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q_sa = 0.0
            for prob, next_state, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * values[next_state] * (1 - done))
            q_values.append(q_sa)
        policy[s] = int(np.argmax(q_values))

    if track_history:
        return policy, values, value_history, max_change_history

    return policy, values, None, None