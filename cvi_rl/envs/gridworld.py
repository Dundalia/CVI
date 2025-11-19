# cvi_rl/envs/gridworld.py

from __future__ import annotations

from typing import Tuple, Optional

import gymnasium as gym
from .base import TabularEnvSpec


def make_gridworld_env(
    *,
    env_id: str = "FrozenLake-v1",
    map_name: str = "4x4",
    is_slippery: bool = False,
    render_mode: Optional[str] = None,
    initial_state: int = 0,
    **env_kwargs,
) -> Tuple[TabularEnvSpec, gym.Env]:
    """
    Create a tabular gridworld-like environment using Gym's FrozenLake.

    Parameters
    ----------
    env_id : str
        Gym ID. Default: "FrozenLake-v1".
    map_name : str
        One of Gym's predefined maps, e.g. "4x4", "8x8".
    is_slippery : bool
        If False, the environment becomes deterministic (good for DP).
    render_mode : str, optional
        E.g. "ansi", "human".
    initial_state : int
        Default starting state index. For FrozenLake this is usually 0.
    env_kwargs : dict
        Extra kwargs forwarded to gym.make.

    Returns
    -------
    spec : TabularEnvSpec
    env : gym.Env
    """
    kwargs = dict(map_name=map_name, is_slippery=is_slippery)
    kwargs.update(env_kwargs)

    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **kwargs)
    P = env.unwrapped.P

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # For FrozenLake, terminal states are those where "done" is True.
    # We can try to infer them from P.
    terminal_states = []
    for s in range(n_states):
        # A state is terminal if all actions from it immediately terminate with prob 1.
        is_terminal = True
        for a in range(n_actions):
            transitions = P[s][a]
            for prob, next_state, reward, done in transitions:
                if not done and prob > 0.0:
                    is_terminal = False
                    break
            if not is_terminal:
                break
        if is_terminal:
            terminal_states.append(s)

    spec = TabularEnvSpec(
        n_states=n_states,
        n_actions=n_actions,
        P=P,
        initial_state=initial_state,
        terminal_states=terminal_states,
        name=f"{env_id}-{map_name}{'-slip' if is_slippery else ''}",
    )

    return spec, env