# cvi_rl/algorithms/tabular_pi.py

#! Determenistic Tabular Policy Iteration

from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
from tqdm import tqdm


def policy_evaluation(
    env_spec: TabularEnvSpec,
    policy: np.ndarray,
    gamma: float,
    termination: float,
    max_iters: int,
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

            #! Update V(s) using Q(s,a) given a DETERMINISTIC policy
            a_pi = int(policy[s])
            v_values[s] = q_values[s, a_pi]

            delta = max(delta, abs(v_old - v_values[s]))

        if delta < float(termination):
            break

    return q_values, v_values


def policy_improvement(
    env_spec: TabularEnvSpec,
    values: np.ndarray,
    gamma: float,
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
    gamma: float,
    eval_termination: float,
    max_policy_eval_iters: int,
    max_policy_iters: int,
    init_policy: Optional[np.ndarray],
    return_v_history: bool,
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
    return_v_history : bool
        If True, also return a list of policies visited.

    Returns
    -------
    policy : np.ndarray
        Final policy (ideally optimal), shape [n_states].
    values : np.ndarray
        Corresponding state-value function V_pi, shape [n_states].
    v_history : list[np.ndarray] or None
        If return_v_history=True, list of V-values over iterations.
    """
    n_states = env_spec.n_states
    n_actions = env_spec.n_actions

    if init_policy is not None:
        policy = np.array(init_policy, dtype=int, copy=True)
    else:
        policy = np.random.randint(0, env_spec.n_actions, size=n_states)

    v_history: Optional[List[np.ndarray]] = [] if return_v_history else None

    for _ in tqdm(range(max_policy_iters), desc="Policy Iteration"):
        # 1) Policy evaluation
        _, v_values = policy_evaluation(
            env_spec,
            policy,
            gamma=gamma,
            termination=eval_termination,
            max_iters=max_policy_eval_iters,
        )
        if return_v_history and v_history is not None:
            v_history.append(v_values.copy())

        # 2) Policy improvement
        new_policy = policy_improvement(env_spec, v_values, gamma=gamma)

        # Check for convergence / stability
        if np.array_equal(new_policy, policy):
            policy = new_policy
            break

        policy = new_policy

    #! Final evaluation with the converged policy
    # _, v_values = policy_evaluation(
    #     env_spec,
    #     policy,
    #     gamma=gamma,
    #     termination=eval_termination,
    #     max_iters=max_policy_eval_iters,
    # )

    if return_v_history:
        return policy, v_values, v_history
    return policy, v_values, None


def run_policy_iteration(env_spec: TabularEnvSpec, env, config: dict, logger=None):
    """
    Run using Policy Iteration algorithm.
    
    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification.
    env : gym.Env
        Gymnasium environment instance (for MC evaluation).
    config : dict
        Training configuration with keys:
        - gamma: float (discount factor)
        - eval_termination: float (convergence threshold for policy evaluation)
        - max_policy_eval_iters: int (max iterations per policy evaluation)
        - max_policy_iters: int (max policy improvement iterations)
        - eval_episodes: int (episodes for MC evaluation, optional)
        - max_steps: int (max steps per episode for MC, optional)
    logger : callable, optional
        Function to log metrics, signature: logger(metrics_dict, step=None)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - policy: Converged policy
        - values: State values V
        - Q_values: Action values Q
        - metrics: Performance metrics
    """
    import time
    import numpy as np
    from .mc import evaluate_policy_monte_carlo
    
    print("\n" + "="*60)
    print("Running Policy Iteration")
    print("="*60)
    
    # Extract config
    gamma = config['gamma']
    eval_termination = config['eval_termination']
    max_policy_eval_iters = config['max_policy_eval_iters']
    max_policy_iters = config['max_policy_iters']
    init_policy = config.get('init_policy', None)
    initial_state = config.get('initial_state', None)
    seed = config.get('seed', None)
    
    print(f"  Policy eval termination: {eval_termination}")
    print(f"  Max policy iterations: {max_policy_iters}")
    print(f"  Gamma: {gamma}")
    
    start_time = time.time()
    
    # Run Policy Iteration
    policy, V_values, v_history = policy_iteration(
        env_spec,
        gamma=gamma,
        eval_termination=eval_termination,
        max_policy_eval_iters=max_policy_eval_iters,
        max_policy_iters=max_policy_iters,
        init_policy=init_policy,
        return_v_history=True
    )
    
    elapsed_time = time.time() - start_time
    
    Q_values, _ = policy_evaluation(
        env_spec,
        policy,
        gamma=gamma,
        termination=eval_termination,
        max_iters=max_policy_eval_iters
    )
    
    metrics = {
        'training_time': elapsed_time,
        'converged_iterations': len(v_history),
        'final_mean_value': float(np.mean(V_values)),
        # 'final_mean_q': float(np.mean(Q_values)),
    }
    
    # Log the value function history
    if logger and v_history is not None:
        for i in range(1, len(v_history)):
            logger({'mean_v_value': float(np.mean(v_history[i]))}, step=i)
            
    
    if config['eval_episodes'] > 0 and env is not None:
        n_episodes = config['eval_episodes']
        max_steps = config['max_steps']
        
        avg_return, var_return, success_rate, _, avg_steps, var_steps = evaluate_policy_monte_carlo(
            env,
            env_spec,
            policy,
            n_episodes=n_episodes,
            gamma=gamma,
            max_steps=max_steps,
            initial_state=initial_state,
            seed=seed
        )
        
        metrics.update({
            'mc_avg_return': float(avg_return),
            'mc_success_rate': float(success_rate),
            # 'mc_var_return': float(var_return),
        })
    
    # Final logging
    if logger:
        logger(metrics)
    
    print(f"\nConverged in {metrics['converged_iterations']} iterations")
    print(f"Training time: {elapsed_time:.2f}s")
    print(f"Mean V: {metrics['final_mean_value']:.3f}")
    # print(f"Mean Q: {metrics['final_mean_q']:.3f}")
    
    return {
        'policy': policy,
        'values': V_values,
        'Q_values': Q_values,
        'metrics': metrics
    }
