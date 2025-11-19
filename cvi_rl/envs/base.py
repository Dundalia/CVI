# cvi_rl/envs/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Protocol, Any

import gymnasium as gym  # preferred in modern setups


# Transition: (probability, next_state, reward, done)
Transition = Tuple[float, int, float, bool]
TransitionModel = Dict[int, Dict[int, List[Transition]]]


@dataclass
class TabularEnvSpec:
    """
    Minimal description of a tabular environment suitable for DP / CVI.

    Attributes
    ----------
    n_states : int
        Number of discrete states.
    n_actions : int
        Number of discrete actions.
    P : dict
        Transition model: P[s][a] = list of (prob, next_state, reward, done).
        This matches the structure of env.unwrapped.P in Taxi/FrozenLake.
    initial_state : int, optional
        Default initial state index (for evaluation / plotting).
    terminal_states : list[int], optional
        Optional list of terminal state indices.
    name : str
        Human-readable name identifier.
    """
    n_states: int
    n_actions: int
    P: TransitionModel
    initial_state: Optional[int] = None
    terminal_states: Optional[List[int]] = None
    name: str = ""


class TabularEnvFactory(Protocol):
    """
    Protocol for environment factory functions used in registry.

    A factory should return (spec, env), where:
      - spec is a TabularEnvSpec
      - env is a gym.Env (or gymnasium.Env) for sampling rollouts
    """

    def __call__(self, **kwargs: Any) -> Tuple[TabularEnvSpec, gym.Env]:
        ...