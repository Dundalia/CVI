import numpy as np
def sample_initial_states(env, eval_episodes) -> np.ndarray:
    N = eval_episodes
    s0 = []
    for _ in range(N):
        s, _ = env.reset()
        s0.append(s)
    return np.array(s0)