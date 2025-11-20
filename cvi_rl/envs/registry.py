# cvi_rl/envs/registry.py

from __future__ import annotations

from typing import Tuple, Any

import gymnasium as gym

from .base import TabularEnvSpec, TabularEnvFactory
from .taxi import make_taxi_env
from .gridworld import make_gridworld_env
from .cliffwalking import make_cliffwalking_env


def make_env(name: str, **kwargs: Any) -> Tuple[TabularEnvSpec, gym.Env]:
    """
    Simple environment factory.

    Parameters
    ----------
    name : str
        Environment name identifier:
          - "taxi" or "taxi-v3"
          - "frozenlake-4x4", "frozenlake-8x8"
          - "gridworld" (alias to FrozenLake 4x4 deterministic)
    kwargs : dict
        Extra parameters forwarded to the specific env maker.

    Returns
    -------
    spec : TabularEnvSpec
    env : gym.Env
    """
    name = name.lower()

    # Taxi aliases
    if name in {"taxi", "taxi-v3"}:
        return make_taxi_env(**kwargs)

    # Handle all FrozenLake variants dynamically
    if name.startswith("frozenlake"):
        # Parse map_name from name or kwargs
        parts = name.split("-")
        map_name = "4x4"  # default
        if len(parts) >= 2:
            map_name = parts[1]
        # Allow override from kwargs, but avoid conflict
        map_name = kwargs.pop('map_name', map_name)
        is_slippery = "slip" in parts or kwargs.pop('is_slippery', False)
        return make_gridworld_env(map_name=map_name, is_slippery=is_slippery, **kwargs)

    # CliffWalking
    if name == "cliffwalking":
        return make_cliffwalking_env(**kwargs)

    raise ValueError(f"Unknown environment name: {name!r}")