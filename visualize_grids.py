#!/usr/bin/env python3
"""
Visualize different omega grid strategies in a compact, publication-quality format.

Usage:
    python visualize_grids.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cvi_rl.cf.grids import make_omega_grid

# Grid parameters
W = 10.0
K = 128

strategies = [
    ("uniform", {}, "Uniform"),
    ("two_density_regions", {}, "Two Density Regions"),
    ("three_density_regions", {}, "Three Density Regions"),
    ("four_density_regions", {}, "Four Density Regions"),
    ("exponential_decay", {"lam": 2.0}, "Exponential Decay"),
    ("linear_decay", {"alpha": 2.0}, "Linear Decay"),
    ("quadratic_decay", {"beta": 3.0}, "Quadratic Decay"),
]

# Color palette - clean, distinct colors
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

def compute_density(omegas, W, n_bins=500):
    """Compute empirical density from grid points."""
    # Create histogram
    hist, bin_edges = np.histogram(omegas, bins=n_bins, range=(-W, W), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


# Create figure with compact layout
fig, ax = plt.subplots(figsize=(10, 7))

# Spacing between grid rows
row_height = 1.0
y_positions = np.arange(len(strategies)) * row_height

# Plot each grid
for idx, (strategy, kwargs, label) in enumerate(strategies):
    y_pos = y_positions[idx]
    color = colors[idx]
    
    # Generate grid
    omegas = make_omega_grid(strategy=strategy, W=W, K=K, **kwargs)
    
    # Compute density PDF
    x_dense = np.linspace(-W, W, 1000)
    bin_centers, density = compute_density(omegas, W, n_bins=100)
    
    # Smooth density for better visualization
    from scipy.ndimage import gaussian_filter1d
    density_smooth = gaussian_filter1d(density, sigma=2)
    
    # Normalize density to fit in row height
    density_norm = density_smooth / density_smooth.max() * (row_height * 0.7)
    
    # Plot filled density curve
    ax.fill_between(bin_centers, y_pos, y_pos + density_norm, 
                     color=color, alpha=0.3, linewidth=0)
    ax.plot(bin_centers, y_pos + density_norm, 
            color=color, linewidth=1.5, alpha=0.8)
    
    # Plot grid points
    ax.scatter(omegas, np.full_like(omegas, y_pos), 
               c=color, s=15, alpha=0.8, zorder=10, edgecolors='white', linewidths=0.5)
    
    # Add horizontal baseline
    ax.axhline(y_pos, color='gray', linewidth=0.5, alpha=0.3, zorder=1)

# Styling
ax.set_xlim(-W * 1.05, W * 1.05)
ax.set_ylim(-0.5, len(strategies) * row_height - 0.3)
ax.set_xlabel('Frequency ω', fontsize=13, fontweight='bold')
ax.set_yticks(y_positions)
ax.set_yticklabels([label for _, _, label in strategies], fontsize=11)

# Add vertical line at origin
ax.axvline(0, color='#2c3e50', linestyle='--', linewidth=1, alpha=0.5, zorder=0)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_color('#2c3e50')

# Remove x-axis ticks on top
ax.tick_params(axis='x', which='both', top=False)
ax.tick_params(axis='y', which='both', left=False, right=False)

# Add title
ax.set_title(f'Frequency Grid Strategies (K={K}, W={W})', 
             fontsize=14, fontweight='bold', pad=15)

# Adjust layout
plt.tight_layout()
plt.savefig('grid_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved visualization to grid_comparison.png")
plt.show()

