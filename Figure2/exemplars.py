import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

sns.set(style="white", rc={"figure.figsize": (6, 6)})

# === Step 1: Paths ===
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent

eq_file = base_dir / "data" / "eq_values.csv"
community_dir = base_dir / "raw_results" / "res_100" / "community_detection"
output_dir = script_dir  # Save images in the Figure2 folder

# === Step 2: Load brain mass data ===
eq_df = pd.read_csv(eq_file)
eq_df = eq_df.dropna(subset=["Brain Mass (g)"])
eq_df_sorted = eq_df.sort_values("Brain Mass (g)")

# Small, medium, large brain mass species
small_species = eq_df_sorted.iloc[0]["Species"]
medium_species = eq_df_sorted.iloc[len(eq_df_sorted)//2]["Species"]
large_species = eq_df_sorted.iloc[-1]["Species"]

species_list = [
    (small_species, "Small Brain"),
    (medium_species, "Medium Brain"),
    (large_species, "Large Brain")
]

# === Step 3: Plot function ===
def plot_communities(species_name, label):
    npy_path = Path(__file__).resolve().parents[2] / "raw_results" / "res_100" / "community_detection" / f"{species}_communities.npy"
    data = np.load(npy_path, allow_pickle=True)

    # Pick the first run
    communities = data[0]

    num_nodes = len(communities)
    num_communities = len(set(communities))

    # Use circular layout
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x, y, c=communities, cmap="tab10", s=100, edgecolor="k", linewidth=0.5)
    plt.title(f"{label}: {species_name}", fontsize=14, pad=15)
    plt.axis("off")
    plt.tight_layout()

    out_path = output_dir / f"Figure2_PanelA_{species_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved: {out_path}")

# === Step 4: Generate all three ===
for species, label in species_list:
    plot_communities(species, label)
