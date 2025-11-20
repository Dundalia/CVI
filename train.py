#!/usr/bin/env python3
"""
Training script for RL algorithms.
Simplified version that delegates training logic to algorithm modules.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Any
import datetime

import yaml

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cvi_rl.envs.registry import make_env


class ConfigLoader:
    """Load and merge configuration from YAML files and command line arguments."""
    
    @staticmethod
    def load_yaml(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config is not None else {}
    
    @staticmethod
    def parse_override(override_str: str) -> tuple[str, Any]:
        """Parse a command line override in the format 'key.subkey=value'."""
        key_path, value_str = override_str.split('=', 1)
        
        try:
            value = yaml.safe_load(value_str)
        except:
            value = value_str
            
        return key_path, value
    
    @staticmethod
    def set_nested(config: Dict, key_path: str, value: Any) -> None:
        """Set a nested dictionary value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @classmethod
    def load(cls, config_path: str, overrides: list[str] = None) -> Dict[str, Any]:
        """Load config from file and apply command line overrides."""
        config = cls.load_yaml(config_path)
        
        if overrides:
            for override in overrides:
                key_path, value = cls.parse_override(override)
                cls.set_nested(config, key_path, value)
        
        return config


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train RL algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py config/monte_carlo.yaml
  python train.py config/policy_eval.yaml logger.do.online=true
        """
    )
    
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in format key.subkey=value'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigLoader.load(args.config, args.overrides)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        return 1
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Extract sections
    env_config = config.get('env', {})
    training_config = config.get('training', {})
    logger_config = config.get('logger', {})
    algorithm = training_config.get('algorithm')
    
    if not algorithm:
        print("Error: No algorithm specified in config (training.algorithm)")
        return 1
    
    # Initialize W&B if configured
    wandb_run = None
    logger_func = None
    
    if WANDB_AVAILABLE and logger_config.get('do', {}).get('online', False):
        run_name = logger_config.get('run_name', None)
        if run_name:
            date_time = datetime.datetime.now().strftime("%d/%m-%H:%M:%S")
            run_name = f"{run_name} {date_time}"
        
        wandb_run = wandb.init(
            project=logger_config.get('project_name', 'CVI-RL'),
            name=run_name,
            config=config,
            tags=logger_config.get('tags', []),
            reinit=True
        )
        logger_func = lambda metrics, step=None: wandb.log(metrics, step=step)
        print(f"W&B logging enabled: {logger_config.get('project_name')}")
    else:
        print("W&B logging disabled")
    
    # Initialize environment
    env_name = env_config.get('name', 'taxi')
    env_kwargs = env_config.get('kwargs', {})
    
    print(f"\nInitializing environment: {env_name}")
    env_spec, env = make_env(env_name, **env_kwargs)
    
    print(f"  States: {env_spec.n_states}")
    print(f"  Actions: {env_spec.n_actions}")
    print(f"  Gamma: {env_config.get('gamma', 0.99)}")
    
    # Get algorithm configuration
    algo_config = training_config.get(algorithm, {})
    
    # Get module and function paths from config
    module_path = algo_config.get('module')
    function_name = algo_config.get('function')
    
    if not module_path or not function_name:
        print(f"Error: Algorithm '{algorithm}' missing 'module' or 'function' in config")
        print(f"Expected: training.{algorithm}.module and training.{algorithm}.function")
        return 1
    
    # Add gamma to algo config
    algo_config['gamma'] = env_config.get('gamma', 0.99)
    
    # Import and run algorithm dynamically
    try:
        # Dynamic import
        import importlib
        module = importlib.import_module(module_path)
        train_func = getattr(module, function_name)
        
        # Run training
        results = train_func(env_spec, env, algo_config, logger_func)
        
        if WANDB_AVAILABLE and wandb_run is not None:
            wandb.summary.update(results['metrics'])
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        
        # Print summary
        if 'metrics' in results:
            print("\nFinal Metrics:")
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if wandb_run:
            wandb.finish()
        if env:
            env.close()


if __name__ == '__main__':
    sys.exit(main())
