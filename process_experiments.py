import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

WANDB_ENTITY = "fatty_data"
WANDB_PROJECT = "CVI-RL"
METRIC_NAME = "td_error"

ALGORITHMS = ["cvi", "pi", "vi", "c51"]
COLORS = {"cvi": "#1f77b4", "pi": "#ff7f0e", "vi": "#2ca02c", "c51": "#d62728"}
ALGO_LABELS = {"cvi": "CVI", "pi": "PI", "vi": "VI", "c51": "C51"}


def get_runs(api, project_path, env_name, algo_name):
    """Fetch runs matching environment and algorithm tags, including all seeds."""
    filters = {
        "tags": {"$all": [f"env_{env_name}", f"algo_{algo_name}"]},
        "state": "finished"
    }
    runs = api.runs(project_path, filters=filters)
    
    # Filter to only include runs that have all three required tag prefixes
    valid_runs = []
    for run in runs:
        tags = run.tags
        has_env = any(tag.startswith("env_") for tag in tags)
        has_algo = any(tag.startswith("algo_") for tag in tags)
        has_seed = any(tag.startswith("seed_") for tag in tags)
        
        if has_env and has_algo and has_seed:
            valid_runs.append(run)
    
    return valid_runs


def process_experiment(env):
    """Process and plot mean V-value for all algorithms on a given environment."""
    print(f"Processing environment: {env}")
    
    api = wandb.Api()
    project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
    
    algo_data = {}  # Store aggregated data per algorithm
    
    for algo in ALGORITHMS:
        runs = get_runs(api, project_path, env, algo)
        
        all_runs_data = []
        for run in runs:
            history = run.history(keys=[METRIC_NAME, "_step"])
            if history.empty or METRIC_NAME not in history.columns:
                continue
            
            h = history[["_step", METRIC_NAME]].dropna().sort_values("_step")
            h["run_id"] = run.id  # Keep track of individual runs
            all_runs_data.append(h)
            print(f"  Found run: {run.name} (algo={algo})")
        
        if all_runs_data:
            # Combine all runs for this algorithm
            combined = pd.concat(all_runs_data, ignore_index=True)
            algo_data[algo] = combined
            
            # Print stats
            min_step = combined.groupby("run_id")["_step"].min().min()
            max_step = combined.groupby("run_id")["_step"].max().max()
            print(f"    {algo}: {len(all_runs_data)} runs, steps {min_step:.0f} to {max_step:.0f}")
    
    if not algo_data:
        print(f"No data found for environment: {env}")
        return
    
    # Find global max step across all runs
    max_step = max(df["_step"].max() for df in algo_data.values())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    final_values = {}
    
    for algo, df in algo_data.items():
        color = COLORS.get(algo, "gray")
        
        # Calculate mean and std for this algorithm across runs
        grouped = df.groupby("_step")[METRIC_NAME].agg(['mean', 'std']).reset_index()
        algo_max_step = grouped["_step"].max()
        final_value = grouped["mean"].iloc[-1]
        final_values[algo] = final_value
        
        # Plot mean line
        ax.plot(grouped["_step"], grouped["mean"], color=color, label=ALGO_LABELS[algo], linewidth=2)
        
        # Plot variance (shaded area)
        ax.fill_between(
            grouped["_step"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            color=color,
            alpha=0.2
        )
        
        # Extend with dotted line if needed
        if algo_max_step < max_step:
            ax.plot(
                [algo_max_step, max_step],
                [final_value, final_value],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7
            )
    
    # # Add final value annotations
    # y_offset = 0
    # for algo in ALGORITHMS:
    #     if algo in final_values:
    #         ax.annotate(
    #             f"{ALGO_LABELS[algo]}: {final_values[algo]:.2f}",
    #             xy=(max_step, final_values[algo]),
    #             xytext=(10, y_offset),
    #             textcoords="offset points",
    #             fontsize=9,
    #             color=COLORS.get(algo, "gray"),
    #             fontweight="bold"
    #         )
    #         y_offset += 12  # Offset to avoid overlapping
    
    ax.set_title(env.capitalize(), fontweight='bold', fontsize=14)
    ax.set_xlabel("Iteration", fontweight='bold', fontsize=12)
    ax.set_ylabel("Value Function Approximation Error", fontweight='bold', fontsize=12)
    ax.legend(title="Algorithm", loc="lower right")
    
    ax.set_yscale('log')
    
    os.makedirs("figures", exist_ok=True)
    output_path = f"figures/td_error_{env}.png"
    plt.savefig(output_path, dpi=125, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    process_experiment(env="taxi")
    process_experiment(env="frozenlake")
    
    
