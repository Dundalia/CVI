# Adding a New Algorithm to the Training System

The training system is designed to be easily extensible. You can add new algorithms without modifying `train.py`.

## Steps to Add a New Algorithm

### 1. Create the Training Function

In your algorithm file (e.g., `cvi_rl/algorithms/my_new_algo.py`), add a training function with this signature:

```python
def train_my_algorithm(env_spec, env, config, logger=None):
    """
    Train using my custom algorithm.
    
    Parameters
    ----------
    env_spec : TabularEnvSpec
        Environment specification
    env : gym.Env
        Gymnasium environment instance
    config : dict
        Algorithm configuration from YAML (includes 'gamma')
    logger : callable, optional
        Function to log metrics: logger(metrics_dict, step=None)
    
    Returns
    -------
    results : dict
        Must include:
        - 'metrics': dict of performance metrics
        - Other algorithm-specific outputs
    """
    import time
    import numpy as np
    
    print("\n" + "="*60)
    print("Running My Custom Algorithm")
    print("="*60)
    
    # Extract config
    gamma = config.get('gamma', 0.99)
    my_param = config.get('my_param', 100)
    
    # Your algorithm logic here
    start_time = time.time()
    
    # ... training code ...
    
    elapsed_time = time.time() - start_time
    
    # Prepare metrics
    metrics = {
        'algorithm': 'my_algorithm',
        'training_time': elapsed_time,
        # ... other metrics ...
    }
    
    # Log to W&B if logger provided
    if logger:
        logger(metrics)
    
    # Print results
    print(f"\nTraining time: {elapsed_time:.2f}s")
    
    return {
        'metrics': metrics,
        # ... other outputs ...
    }
```

### 2. Add Configuration to YAML

In your config file (e.g., `config/my_experiment.yaml`):

```yaml
env:
  name: taxi
  gamma: 0.99
  kwargs: {}

training:
  algorithm: my_algo  # <-- Choose a name
  
  # Algorithm configuration
  my_algo:
    module: cvi_rl.algorithms.my_new_algo  # <-- Python import path
    function: train_my_algorithm            # <-- Function name
    my_param: 100                           # <-- Your custom parameters
    another_param: 0.5

logger:
  do:
    online: false
  project_name: MyProject
  run_name: my_experiment
  tags:
    - my_algo
```

### 3. Run Your Algorithm

```bash
python train.py config/my_experiment.yaml
```

That's it! No need to modify `train.py` at all.

## Example: Adding Value Iteration

**File: `cvi_rl/algorithms/tabular_vi.py`**

Add this function:

```python
def train_value_iteration(env_spec, env, config, logger=None):
    """Train using Value Iteration."""
    import time
    import numpy as np
    
    print("\n" + "="*60)
    print("Running Value Iteration")
    print("="*60)
    
    gamma = config.get('gamma', 0.99)
    max_iterations = config.get('max_iterations', 1000)
    termination = config.get('termination', 1e-4)
    
    start_time = time.time()
    
    policy, values, _, _ = value_iteration(
        env_spec,
        gamma=gamma,
        iterations=max_iterations,
        termination=termination,
        track_history=False
    )
    
    elapsed_time = time.time() - start_time
    
    metrics = {
        'algorithm': 'value_iteration',
        'training_time': elapsed_time,
        'mean_value': float(np.mean(values)),
    }
    
    if logger:
        logger(metrics)
    
    print(f"Training time: {elapsed_time:.2f}s")
    
    return {'policy': policy, 'values': values, 'metrics': metrics}
```

**Config: `config/value_iteration.yaml`**

```yaml
env:
  name: taxi
  gamma: 0.99

training:
  algorithm: vi
  
  vi:
    module: cvi_rl.algorithms.tabular_vi
    function: train_value_iteration
    max_iterations: 1000
    termination: 1e-4

logger:
  do:
    online: false
```

**Run it:**

```bash
python train.py config/value_iteration.yaml
```

## Benefits

✅ **No code changes to `train.py`** when adding algorithms  
✅ **Each algorithm is self-contained** in its own module  
✅ **Easy to experiment** with different algorithms via config files  
✅ **Clear separation of concerns** - train.py handles infrastructure, algorithms handle logic  
✅ **Type-safe** - all algorithms follow the same interface  

## Config Parameters Available

Each algorithm's `config` dict automatically includes:
- `gamma`: Discount factor (from `env.gamma`)
- All parameters under `training.{algorithm}` in the YAML
- `module` and `function` are used for import and then removed from config

## Logging

If you want to log metrics to W&B:

```python
if logger:
    # Log during training
    for iteration in range(n_iters):
        logger({'iteration': iteration, 'loss': loss}, step=iteration)
    
    # Log final metrics
    logger(final_metrics)
```

The logger is only provided if W&B is enabled in the config (`logger.do.online: true`).
