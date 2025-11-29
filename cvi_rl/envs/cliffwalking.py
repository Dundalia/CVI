from __future__ import annotations

import gymnasium as gym

import numpy as np

from .base import TabularEnvSpec, TransitionModel


def make_cliffwalking_env(**kwargs) -> tuple[TabularEnvSpec, gym.Env]:
    """
    Create a CliffWalking environment.

    CliffWalking is a 4x12 grid world where the agent must navigate from start to goal
    while avoiding cliffs that incur a large negative reward. The environment is stochastic
    by default (slippery actions).

    Parameters
    ----------
    **kwargs
        Passed to gym.make("CliffWalking-v0").

    Returns
    -------
    spec : TabularEnvSpec
    env : gym.Env
    """
    env = gym.make("CliffWalking-v1", **kwargs)

    # Extract tabular info
    nS = env.observation_space.n
    nA = env.action_space.n
    P = env.unwrapped.P

    spec = TabularEnvSpec(
        n_states=nS,
        n_actions=nA,
        P=P,  
        initial_state=0,  # Standardized initial state for fair comparison
        terminal_states=None,  # CliffWalking termination encoded via done in P
        name="cliffwalking",
    )

    return spec, env