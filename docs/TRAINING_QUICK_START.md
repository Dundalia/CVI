# Training System - Quick Reference

## Architecture

```
train.py (170 lines)
  ├─ Loads YAML config
  ├─ Initializes environment
  ├─ Sets up W&B (if enabled)
  └─ Dynamically imports & calls algorithm.train_*()

Algorithm files (e.g., mc.py, tabular_pi.py)
  └─ Contains train_*() function with all algorithm logic
```

## Running Training

```bash
# Basic usage
python train.py config/taxi.yaml

# Override parameters
python train.py config/taxi.yaml training.mc.n_episodes=5000

# Enable W&B
python train.py config/taxi.yaml logger.do.online=true logger.run_name=exp_01
```

## Config Structure

```yaml
env:
  name: taxi
  gamma: 0.99
  kwargs: {}

training:
  algorithm: mc  # Which algorithm to use
  
  mc:  # Algorithm-specific config
    module: cvi_rl.algorithms.mc      # Import path
    function: train_monte_carlo        # Function name
    n_episodes: 1000                   # Custom params
    max_steps: 200

logger:
  do:
    online: false
  project_name: CVI-RL
  run_name: null
  tags: []
```

## Currently Supported Algorithms

| Algorithm | Config Key | Module | Function |
|-----------|------------|--------|----------|
| Monte Carlo | `mc` | `cvi_rl.algorithms.mc` | `train_monte_carlo` |
| Policy Evaluation | `policy_eval` | `cvi_rl.algorithms.tabular_pi` | `train_policy_evaluation` |
| Policy Iteration | `policy_iter` | `cvi_rl.algorithms.tabular_pi` | `train_policy_iteration` |

## Adding New Algorithms

See `docs/ADDING_ALGORITHMS.md` for detailed instructions.

**Quick steps:**
1. Add `train_*()` function to algorithm file
2. Add config section to YAML with `module` and `function`
3. Run with `python train.py your_config.yaml`

**No changes to `train.py` needed!** ✨
