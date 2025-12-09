#!/usr/bin/env python3
"""
Visualize different omega grid strategies.

Usage:
    python visualize_grids.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from cvi_rl.cf.grids import make_omega_grid

# Grid parameters
W = 10.0
K = 64

strategies = [
    ("uniform", {}),
    ("two_density_regions", {}),
    ("three_density_regions", {}),
    ("four_density_regions", {}),
    ("exponential_decay", {"lam": 2.0}),
    ("linear_decay", {"alpha": 2.0}),
    ("quadratic_decay", {"beta": 3.0}),
]

fig, axes = plt.subplots(len(strategies), 1, figsize=(12, 2.5 * len(strategies)))

for idx, (strategy, kwargs) in enumerate(strategies):
    ax = axes[idx]
    
    # Generate grid
    omegas = make_omega_grid(strategy=strategy, W=W, K=K, **kwargs)
    
    # Plot points on horizontal axis
    ax.scatter(omegas, np.zeros_like(omegas), alpha=0.6, s=20)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Set limits and labels
    ax.set_xlim(-W * 1.1, W * 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('ω (frequency)', fontsize=10)
    ax.set_yticks([])
    ax.set_title(f'{strategy.upper()} (K={K})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Add statistics
    near_zero = np.sum(np.abs(omegas) < 1.0)
    avg_spacing_center = np.mean(np.diff(omegas[K//2-2:K//2+2]))
    
    stats_text = f'Points within [-1,1]: {near_zero} | Avg spacing near 0: {avg_spacing_center:.4f}'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('grid_comparison.png', dpi=150, bbox_inches='tight')
print("Saved visualization to grid_comparison.png")
plt.show()

