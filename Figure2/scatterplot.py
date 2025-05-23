import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# === Paths ===
root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "eq_values.csv"
npy_dir = root.parent / "raw_results" / "res_100" / "community_detection"
output_path = Path(__file__).resolve().parent / "Figure2_PanelB_Communities_vs_BrainMass.png"

# === Load EQ Data ===
eq_df = pd.read_csv(data_path)
eq_df.dropna(subset=["Species", "Brain Mass (g)"], inplace=True)

# === Compute Communities ===
records = []

for _, row in eq_df.iterrows():
    species = row["Species"]
    brain_mass = row["Brain Mass (g)"]

    npy_file = npy_dir / f"{species}_communities.npy"
    if not npy_file.exists():
        continue

    data = np.load(npy_file, allow_pickle=True)
    if data.ndim != 2:
        continue

    first_run = data[0]
    num_communities = len(np.unique(first_run))

    records.append({
        "Species": species,
        "Brain Mass (g)": brain_mass,
        "# Communities": num_communities
    })

# === Create DataFrame and Plot ===
df = pd.DataFrame(records)

# Prepare log-log values
df["log_brain_mass"] = np.log10(df["Brain Mass (g)"])
df["log_communities"] = np.log10(df["# Communities"])

# Linear regression in log-log space
slope, intercept, r_value, p_value, _ = linregress(df["log_brain_mass"], df["log_communities"])
line_x = np.linspace(df["log_brain_mass"].min(), df["log_brain_mass"].max(), 100)
line_y = slope * line_x + intercept

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df["Brain Mass (g)"], df["# Communities"], s=60, edgecolor='k', alpha=0.7)
plt.plot(10**line_x, 10**line_y, 'r--', label=f'log-log fit: y ~ x^{slope:.2f}, $R^2$={r_value**2:.2f}')

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Brain Mass (g, log scale)")
plt.ylabel("Number of Functional Communities (log scale)")
plt.title("Panel B: Brain Mass vs. Number of Functional Communities")
plt.legend()
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved Panel B with regression to: {output_path}")
