# cvi_rl/envs/taxi.py

from __future__ import annotations

from typing import Tuple, Optional

import gymnasium as gym
from .base import TabularEnvSpec


# ----- Helpers to encode / decode Taxi states -----
def encode_state(
    taxi_row: int,
    taxi_col: int,
    passenger_loc: int,
    destination: int,
) -> int:
    """
    Encode a Taxi-v3 state from its components.

    passenger_loc: 0=R, 1=G, 2=Y, 3=B, 4=in_taxi
    destination:   0=R, 1=G, 2=Y, 3=B
    """
    return taxi_row * 100 + taxi_col * 20 + passenger_loc * 4 + destination


def decode_state(state: int):
    """Decode a Taxi-v3 state into (taxi_row, taxi_col, passenger_loc, destination)."""
    destination = state % 4
    state //= 4
    passenger_loc = state % 5
    state //= 5
    taxi_col = state % 5
    taxi_row = state // 5
    return taxi_row, taxi_col, passenger_loc, destination


def default_taxi_initial_state() -> int:
    """
    Choose a reasonable default starting configuration.

    Example: taxi at (2, 0), passenger at R (0), destination Y (2).
    """
    taxi_row = 2
    taxi_col = 0
    passenger_loc = 0  # Red
    destination = 2    # Yellow
    return encode_state(taxi_row, taxi_col, passenger_loc, destination)


def make_taxi_env(
    *,
    env_id: str = "Taxi-v3",
    render_mode: Optional[str] = None,
    initial_state: Optional[int] = None,
    **env_kwargs,
) -> Tuple[TabularEnvSpec, gym.Env]:
    """
    Create a Taxi environment and its TabularEnvSpec.

    Parameters
    ----------
    env_id : str
        Gym ID for the Taxi environment. Usually "Taxi-v3".
    render_mode : str, optional
        E.g. "ansi", "human", etc. Passed to gym.make.
    initial_state : int, optional
        If provided, use this as default initial state; otherwise use a
        hard-coded reasonable default via default_taxi_initial_state().
    env_kwargs : dict
        Extra kwargs forwarded to gym.make (e.g., is_rainy=True, fickle_passenger=True
        if your custom Taxi env supports them).

    Returns
    -------
    spec : TabularEnvSpec
    env : gym.Env
    """
    # Explicitly pass render_mode via env_kwargs if given
    if render_mode is not None:
        env_kwargs = dict(env_kwargs)  # copy
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)
    # In both gym and gymnasium, P is typically on the unwrapped env
    P = env.unwrapped.P

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    if initial_state is None:
        initial_state = default_taxi_initial_state()

    spec = TabularEnvSpec(
        n_states=n_states,
        n_actions=n_actions,
        P=P,
        initial_state=initial_state,
        terminal_states=None,  # Taxi termination encoded via done in P
        name="taxi-v3",
    )

    return spec, env