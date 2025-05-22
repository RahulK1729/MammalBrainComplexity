import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths (relative to script location)
data_dir = Path(__file__).resolve().parents[1] / "data"
output_dir = Path(__file__).resolve().parent

# Load data
eq_df = pd.read_csv(data_dir / "eq_values.csv")
gi_df = pd.read_csv(data_dir / "gi_values.csv")

# Merge datasets on Species
merged_df = pd.merge(eq_df, gi_df, on="Species", suffixes=('_eq', '_gi'))

# === Panel A: Brain Mass vs EQ ===
fig_a, ax1 = plt.subplots(figsize=(7, 6))
ax1.scatter(merged_df["Brain Mass (g)"], merged_df["EQ Value"], alpha=0.7)
ax1.set_xlabel("Brain Mass (g)")
ax1.set_ylabel("Encephalization Quotient (EQ)")
ax1.set_title("Panel A: Brain Mass vs. EQ")
ax1.set_xscale("log")
ax1.grid(True)
fig_a.tight_layout()
fig_a.savefig(output_dir / "Figure1_PanelA.png", dpi=300)

# === Panel B: Brain Mass vs GI ===
fig_b, ax2 = plt.subplots(figsize=(7, 6))
ax2.scatter(merged_df["Brain Mass (g)"], merged_df["GI"], alpha=0.7, color='orange')
ax2.set_xlabel("Brain Mass (g)")
ax2.set_ylabel("Gyrification Index (GI)")
ax2.set_title("Panel B: Brain Mass vs. GI")
ax2.set_xscale("log")
ax2.grid(True)
fig_b.tight_layout()
fig_b.savefig(output_dir / "Figure1_PanelB.png", dpi=300)

plt.show()

print("Saved:")
print(f"- {output_dir / 'Figure1_PanelA.png'}")
print(f"- {output_dir / 'Figure1_PanelB.png'}")
