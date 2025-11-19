# cvi_rl/envs/registry.py

from __future__ import annotations

from typing import Tuple, Any

import gymnasium as gym

from .base import TabularEnvSpec, TabularEnvFactory
from .taxi import make_taxi_env
from .gridworld import make_gridworld_env


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

    # Gridworld / FrozenLake variants
    if name in {"gridworld", "frozenlake-4x4"}:
        # default: small deterministic gridworld
        return make_gridworld_env(map_name="4x4", is_slippery=False, **kwargs)

    if name in {"frozenlake-8x8"}:
        return make_gridworld_env(map_name="8x8", is_slippery=False, **kwargs)

    # If user passes a full env_id we don't recognize as a shortcut:
    #   e.g., make_env("FrozenLake-v1", map_name="4x4")
    if name.startswith("frozenlake"):
        # Try to parse map_name from name if provided, else fall back to kwargs
        # Example: "frozenlake-4x4-slip"
        parts = name.split("-")
        map_name = "4x4"
        if len(parts) >= 2:
            map_name = parts[1]
        is_slippery = "slip" in parts or kwargs.get("is_slippery", False)
        return make_gridworld_env(map_name=map_name, is_slippery=is_slippery, **kwargs)

    raise ValueError(f"Unknown environment name: {name!r}")