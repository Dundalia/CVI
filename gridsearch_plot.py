import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("cvi_gridsearch_taxi.csv")

# Rename labels for publication
GRID_LABELS = {
    "uniform": "Uniform",
    "two_density_regions": "Two-Density",
    "three_density_regions": "Three-Density",
    "four_density_regions": "Four-Density",
    "exponential_decay": "Exponential",
    "linear_decay": "Linear Decay",
    "quadratic_decay": "Quadratic Decay"
}

INTERP_LABELS = {
    "linear": "Linear",
    "polar": "Polar",
    "pchip": "PCHIP",
    "lanczos": "Lanczos"
}

COLLAPSE_LABELS = {
    "ls": "Least Squares",
    "fft": "FFT",
    "gaussian": "Gaussian"
}

# Apply label mappings
df["grid_strategy"] = df["grid_strategy"].map(GRID_LABELS)
df["interp_method"] = df["interp_method"].map(INTERP_LABELS)
df["collapse_method"] = df["collapse_method"].map(COLLAPSE_LABELS)

# Summary statistics by method
print("=" * 60)
print("AVERAGE MAE BY METHOD")
print("=" * 60)

print("\n📊 By Grid Strategy:")
print(df.groupby("grid_strategy")["mae"].mean().sort_values().to_string())

print("\n📊 By Interpolation Method:")
print(df.groupby("interp_method")["mae"].mean().sort_values().to_string())

print("\n📊 By Collapse Method:")
print(df.groupby("collapse_method")["mae"].mean().sort_values().to_string())

print("\n📊 By K (atoms):")
print(df.groupby("K")["mae"].mean().sort_values().to_string())

# Best combinations
print("\n" + "=" * 60)
print("TOP 10 BEST CONFIGURATIONS (lowest MAE)")
print("=" * 60)
best = df.nsmallest(10, "mae")[["grid_strategy", "interp_method", "collapse_method", "K", "mae", "rmse"]]
print(best.to_string(index=False))

# Create compact heatmap plot
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Heatmap 1: Grid Strategy vs Collapse Method
pivot1 = df.pivot_table(values="mae", index="grid_strategy", columns="collapse_method", aggfunc="mean")
sns.heatmap(pivot1, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[0], cbar_kws={'label': 'MAE'})
axes[0].set_title("Grid Strategy vs Collapse Method")
axes[0].set_xlabel("Collapse Method")
axes[0].set_ylabel("Grid Strategy")

# Heatmap 2: Interp Method vs Collapse Method
pivot2 = df.pivot_table(values="mae", index="interp_method", columns="collapse_method", aggfunc="mean")
sns.heatmap(pivot2, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[1], cbar_kws={'label': 'MAE'})
axes[1].set_title("Interpolation vs Collapse Method")
axes[1].set_xlabel("Collapse Method")
axes[1].set_ylabel("Interpolation Method")

# Heatmap 3: K vs Collapse Method
pivot3 = df.pivot_table(values="mae", index="K", columns="collapse_method", aggfunc="mean")
sns.heatmap(pivot3, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=axes[2], cbar_kws={'label': 'MAE'})
axes[2].set_title("Number of Atoms (K) vs Collapse Method")
axes[2].set_xlabel("Collapse Method")
axes[2].set_ylabel("K")

plt.tight_layout()
plt.savefig("figures/gridsearch_heatmaps.png", dpi=300, bbox_inches="tight")
plt.show()
print("\n✅ Saved heatmaps to figures/gridsearch_heatmaps.png")
