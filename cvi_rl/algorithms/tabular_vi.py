from __future__ import annotations
from typing import Tuple, Optional, List
import numpy as np
from cvi_rl.envs.base import TabularEnvSpec, TransitionModel
import time
import numpy as np
from cvi_rl.algorithms.mc import evaluate_policy_monte_carlo
from cvi_rl.algorithms.utils import sample_initial_states


def value_iteration(
    env_spec: TabularEnvSpec,
    gamma: float,
    iterations: int,
    termination: float,
    track_history: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]], Optional[List[float]]]:
    """
    Classical tabular Value Iteration for a TabularEnvSpec.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification (n_states, n_actions, P, ...).
    gamma : float
        Discount factor in [0,1).
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

    values = np.zeros(n_states, dtype=float)
    policy = np.zeros(n_states, dtype=int)

    value_history: Optional[List[np.ndarray]] = [] if track_history else None
    max_change_history: Optional[List[float]] = [] if track_history else None

    for k in range(iterations):
        if track_history and value_history is not None:
            value_history.append(values.copy())

        max_update = 0.0

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

        if max_update < float(termination):
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

def run_value_iteration(env_spec: TabularEnvSpec, env, config: dict, logger=None):
    """
    Wrapper to run value iteration with specified parameters.

    Parameters
    ----------
    env_spec : TabularEnvSpec
        Tabular environment specification.
    gamma : float
        Discount factor.
    max_iters : int
        Maximum number of iterations.
    termination : float
        Termination threshold.
    track_history : bool
        Whether to track history.

    Returns
    -------
    policy : np.ndarray
        Final greedy policy.
    values : np.ndarray
        Final value function.
    value_history : list[np.ndarray] or None
        History of value functions if tracked.
    max_change_history : list[float] or None
        History of max changes if tracked.
    """
    print("\n" + "="*60)
    print("Running Value Iteration")
    print("="*60)
    
    gamma = config['gamma']
    eval_termination = config['eval_termination']
    max_policy_iters = config['max_policy_iters']
    seed = config.get('seed', None)
    
    print(f"  Value iteration termination: {eval_termination}")
    print(f"  Max value iteration iterations: {max_policy_iters}")
    print(f"  Gamma: {gamma}")
    
    start_time = time.time()
    
    policy, V_values, v_history, max_change_history = value_iteration(
        env_spec,
        gamma,
        iterations=max_policy_iters,
        termination=eval_termination,
        track_history=True,
    )
    
    elapsed_time = time.time() - start_time
    
    states_to_evaluate = sample_initial_states(env, config['eval_episodes'])
    expected_v_from_initial_states = float(np.mean(V_values[states_to_evaluate]))
    
    print(f"First Evaluation: {np.mean(np.mean(v_history[0]))}")

    
    metrics = {
        'training_time': elapsed_time,
        'converged_iterations': len(v_history),
        'expected_initial_state_value': expected_v_from_initial_states,
        "final_mean_v_value": float(np.mean(V_values))
    }
    
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
            states_to_evaluate=states_to_evaluate,
            gamma=gamma,
            max_steps=max_steps,
            seed=seed
        )
        
        metrics.update({
            'mc_avg_return': float(avg_return),
            'mc_success_rate': float(success_rate),
        })
        
    if logger:
        logger(metrics)
    
    print(f"\nConverged in {metrics['converged_iterations']} iterations")
    print(f"Training time: {elapsed_time:.2f}s")
    
    return {
        'policy': policy,
        'values': V_values,
        'metrics': metrics
    }
        