# cvi_rl/algorithms/tabular_pi.py

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from cvi_rl.envs.base import TabularEnvSpec, TransitionModel


def policy_evaluation(
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    gamma: float = 0.9,
    termination: float = 1e-2,
    max_iters: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterative policy evaluation for a fixed tabular policy π.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification.
    policy : np.ndarray
        Policy mapping state -> action, shape [n_states].
    gamma : float
        Discount factor.
    termination : float
        Threshold on max change in V for convergence.
    max_iters : int
        Safety cap on number of iterations.

    Returns
    -------
    q_values : np.ndarray
        Action-value function Q_pi(s,a), shape [n_states, n_actions].
    v_values : np.ndarray
        State-value function V_pi(s), shape [n_states].
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P

    v_values = np.zeros(n_states, dtype=float)
    q_values = np.zeros((n_states, n_actions), dtype=float)

    for _ in range(max_iters):
        delta = 0.0

        for s in range(n_states):
            v_old = v_values[s]

            # Compute Q_pi(s,a) for all actions
            for a in range(n_actions):
                q_sa = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * v_values[next_state] * (1 - done))
                q_values[s, a] = q_sa

            # Use only the action prescribed by the policy for V_pi(s)
            a_pi = int(policy[s])
            v_values[s] = q_values[s, a_pi]

            delta = max(delta, abs(v_old - v_values[s]))

        if delta < termination:
            break

    return q_values, v_values


def policy_improvement(
    env_spec: TabularEnvSpec,
    values: np.ndarray,
    gamma: float = 0.9,
) -> np.ndarray:
    """
    Policy improvement step: given V(s), compute a greedy policy.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment spec (provides P, n_states, n_actions).
    values : np.ndarray
        State-value function V(s), shape [n_states].
    gamma : float
        Discount factor.

    Returns
    -------
    policy : np.ndarray
        Greedy policy mapping state -> action, shape [n_states].
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions
    P: TransitionModel = env_spec.P

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = []

        for a in range(n_actions):
            q_sa = 0.0
            for prob, next_state, reward, done in P[s][a]:
                q_sa += prob * (reward + gamma * values[next_state] * (1 - done))
            q_values.append(q_sa)

        policy[s] = int(np.argmax(q_values))

    return policy


def policy_iteration(
    env_spec: TabularEnvSpec,
    gamma: float = 0.9,
    eval_termination: float = 1e-2,
    max_policy_eval_iters: int = 10_000,
    max_policy_iters: int = 100,
    init_policy: Optional[np.ndarray] = None,
    return_history: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Classic Policy Iteration:
      1. Policy Evaluation (via iterative evaluation)
      2. Policy Improvement
    until policy stabilizes or max_policy_iters is reached.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Environment specification.
    gamma : float
        Discount factor.
    eval_termination : float
        Convergence threshold for policy evaluation.
    max_policy_eval_iters : int
        Max iterations for each policy evaluation phase.
    max_policy_iters : int
        Max number of policy improvement steps.
    init_policy : np.ndarray, optional
        Initial policy. If None, start with uniform zeros (e.g., always action 0).
    return_history : bool
        If True, also return a list of policies visited.

    Returns
    -------
    policy : np.ndarray
        Final policy (ideally optimal), shape [n_states].
    values : np.ndarray
        Corresponding state-value function V_pi, shape [n_states].
    policy_history : list[np.ndarray] or None
        If return_history=True, list of policies over iterations.
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions

    if init_policy is None:
        # Start with a simple default: always take action 0
        policy = np.zeros(n_states, dtype=int)
    else:
        policy = np.array(init_policy, dtype=int, copy=True)

    policy_history: Optional[List[np.ndarray]] = [] if return_history else None

    for _ in range(max_policy_iters):
        if return_history and policy_history is not None:
            policy_history.append(policy.copy())

        # 1) Policy evaluation
        _, v_values = policy_evaluation(
            env_spec,
            policy,
            gamma=gamma,
            termination=eval_termination,
            max_iters=max_policy_eval_iters,
        )

        # 2) Policy improvement
        new_policy = policy_improvement(env_spec, v_values, gamma=gamma)

        # Check for convergence
        if np.array_equal(new_policy, policy):
            policy = new_policy
            break

        policy = new_policy

    # Final evaluation with the converged policy (optional but neat)
    _, v_values = policy_evaluation(
        env_spec,
        policy,
        gamma=gamma,
        termination=eval_termination,
        max_iters=max_policy_eval_iters,
    )

    if return_history:
        return policy, v_values, policy_history

    return policy, v_values, None