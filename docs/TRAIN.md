# Training Script Guide

This guide explains how to use the `train.py` script to run various RL algorithms with configurable parameters and W&B logging.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default configuration
python train.py

# Run with a specific config file
python train.py config/frozen_lake_4x4.yaml

# Override specific parameters
python train.py config/taxi.yaml training.algorithm=cvi env.gamma=0.95

# Enable W&B logging
python train.py config/taxi.yaml logger.do.online=true logger.run_name=my_experiment
```

## Supported Algorithms

The training script supports the following algorithms:

1. **Value Iteration (vi)**: Classic dynamic programming algorithm
2. **Policy Iteration (pi)**: Alternates between policy evaluation and improvement
3. **Policy Evaluation (policy_eval)**: Evaluates a given policy
4. **Monte Carlo (mc)**: Evaluates policy using Monte Carlo rollouts
5. **Characteristic Value Iteration (tabular_cvi/cvi)**: Novel CVI algorithm

## Configuration

Configuration is managed through YAML files. The default `config.yaml` contains all available options.

### Configuration Structure

```yaml
env:
  name: taxi  # Environment name
  gamma: 0.99  # Discount factor
  kwargs: {}  # Environment-specific arguments

training:
  algorithm: vi  # Algorithm to run
  
  # Algorithm-specific settings
  vi:
    max_iterations: 1000
    termination: 1e-4
  
  cvi:
    grid_strategy: uniform
    W: 10.0
    K: 256
    interp_method: linear
    collapse_method: ls
    # ... more CVI parameters

logger:
  do:
    online: false  # Enable/disable W&B logging
  project_name: CVI-RL
  run_name: null
  tags: []
```

### Environment Options

- `taxi`: Taxi-v3 environment
- `gridworld`, `frozenlake-4x4`: Small 4x4 gridworld
- `frozenlake-8x8`: Larger 8x8 gridworld

### CVI-Specific Parameters

#### Grid Strategy
- `uniform`: Uniform spacing in frequency domain
- `piecewise_centered`: Dense center, sparse edges
- `logarithmic`: Logarithmic spacing
- `chebyshev`: Chebyshev nodes
- `adaptive`: Adaptive grid based on reward distribution

#### Interpolation Methods
- `linear`: Linear interpolation
- `polar`: Polar coordinate interpolation (for complex values)
- `pchip`: Piecewise Cubic Hermite Interpolating Polynomial
- `lanczos`: Lanczos kernel interpolation

#### Collapse Methods
- `ls`: Least squares polynomial fit
- `fft`: Fast Fourier Transform inversion
- `gaussian`: Gaussian moment estimation

## Example Configs

The `config/` directory contains several pre-configured examples:

### 1. FrozenLake 4x4 (Value Iteration)
```bash
python train.py config/frozen_lake_4x4.yaml
```

### 2. Taxi (CVI with PCHIP interpolation)
```bash
python train.py config/taxi.yaml
```

### 3. GridWorld (Value Iteration)
```bash
python train.py config/gridworld.yaml
```

### 4. CVI Experiment (Adaptive grid + Lanczos)
```bash
python train.py config/cvi_experiment.yaml
```

## Command Line Overrides

You can override any configuration parameter from the command line:

```bash
# Change algorithm
python train.py config.yaml training.algorithm=cvi

# Change environment and gamma
python train.py config.yaml env.name=gridworld env.gamma=0.95

# CVI-specific overrides
python train.py config/taxi.yaml \
    training.cvi.grid_strategy=adaptive \
    training.cvi.K=512 \
    training.cvi.interp_method=lanczos

# Enable W&B with custom run name
python train.py config.yaml \
    logger.do.online=true \
    logger.project_name=MyProject \
    logger.run_name=experiment_001
```

## Weights & Biases (W&B) Logging

### Setup

1. Install wandb: `pip install wandb`
2. Login: `wandb login`
3. Set your API key (or use environment variable):
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

### Enable Logging

In your config file:
```yaml
logger:
  do:
    online: true
  project_name: CVI-RL
  run_name: my_experiment_name
  tags:
    - experiment
    - cvi
```

Or via command line:
```bash
python train.py config.yaml logger.do.online=true logger.run_name=exp_001
```

### Logged Metrics

The script logs various metrics depending on the algorithm:

**Value Iteration:**
- `iteration`: Iteration number
- `max_delta`: Maximum value change
- `mean_value`, `max_value`, `min_value`: Value function statistics
- `final_mean_q`: Final Q-value mean
- `mc_avg_return`, `mc_success_rate`: Monte Carlo evaluation metrics

**CVI:**
- All VI metrics plus:
- `mae`, `mse`, `rmse`, `max_error`: Error metrics vs baseline
- `final_mean_q_cvi`, `final_mean_q_baseline`: Q-value comparison
- Configuration parameters (grid_strategy, K, W, etc.)

## VS Code Launch Configurations

The `.vscode/launch.json` file contains several pre-configured debug configurations:

1. **Train: FrozenLake 4x4 (VI, Online W&B)**: Value Iteration with W&B
2. **Train: Taxi (CVI, Online W&B)**: CVI on Taxi with W&B
3. **Train: Default Config (Offline)**: Run with default config, no W&B
4. **Train: Value Iteration (Taxi, Offline)**: VI on Taxi
5. **Train: Policy Iteration (GridWorld, Offline)**: PI on GridWorld
6. **Train: CVI Experiment (Online W&B)**: Advanced CVI configuration

To use:
1. Open VS Code
2. Go to Run and Debug (Cmd+Shift+D)
3. Select a configuration from the dropdown
4. Press F5 to start debugging

## Output

The script prints:
- Environment information (states, actions, gamma)
- Algorithm configuration
- Training progress
- Final metrics summary

Example output:
```
Initializing environment: taxi
  States: 500
  Actions: 6
  Gamma: 0.95

============================================================
Running Characteristic Value Iteration (CVI)
============================================================
Computing baseline policy with VI...

CVI Configuration:
  Grid: piecewise_centered, W=20.0, K=512
  Interpolation: pchip
  Collapse: ls

Training time: 2.34s
Mean Q (CVI): 8.234
Mean Q (Baseline): 8.241
MAE: 0.002145
RMSE: 0.003521

============================================================
Training completed successfully!
============================================================

Final Metrics:
  mae: 0.002145
  rmse: 0.003521
  mc_avg_return: 7.823
  mc_success_rate: 0.94
```

## Advanced Usage

### Custom Initial Policy

To evaluate a specific policy (e.g., for policy evaluation), modify the trainer code or create a custom config with policy specifications.

### Grid Search

For hyperparameter tuning, create a script that loops over configs:

```python
import subprocess

grid_strategies = ['uniform', 'piecewise_centered', 'adaptive']
interp_methods = ['linear', 'pchip', 'lanczos']

for grid in grid_strategies:
    for interp in interp_methods:
        subprocess.run([
            'python', 'train.py',
            'config/taxi.yaml',
            f'training.cvi.grid_strategy={grid}',
            f'training.cvi.interp_method={interp}',
            'logger.do.online=true',
            f'logger.run_name=grid_{grid}_interp_{interp}'
        ])
```

### Batch Experiments

Use the notebook `experiment_suite.ipynb` for comprehensive grid searches over CVI hyperparameters.

## Troubleshooting

**Issue: "Config file not found"**
- Make sure you're running from the project root directory
- Check the config file path is correct

**Issue: "W&B not logging"**
- Verify `logger.do.online=true` in config
- Check W&B API key is set correctly
- Ensure wandb is installed: `pip install wandb`

**Issue: "Unknown algorithm"**
- Check spelling of algorithm name
- Valid options: vi, pi, policy_eval, mc, tabular_cvi, cvi

**Issue: CVI numerical errors**
- Try reducing W (frequency range)
- Increase K (grid points)
- Use more stable interpolation (linear instead of lanczos)
- Adjust collapse method parameters

## Contributing

When adding new algorithms:
1. Add the algorithm implementation to `cvi_rl/algorithms/`
2. Add a training method to the `Trainer` class in `train.py`
3. Register it in the `algorithm_map` dictionary
4. Add default config parameters to `config.yaml`
5. Create an example config in `config/`

## References

- CVI Paper: [Add reference]
- Gymnasium Documentation: https://gymnasium.farama.org/
- W&B Documentation: https://docs.wandb.ai/
