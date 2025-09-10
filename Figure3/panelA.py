import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.patches import Ellipse

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure3")
os.makedirs(fig_dir, exist_ok=True)
output_img = os.path.join(fig_dir, "figure3_panelA_morphospace.png")
output_pca_img = os.path.join(fig_dir, "figure3_panelA_PCA_validation.png")

# Load datasets
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))

# Clean species names
for df in [gi_df, eq_df, fc_df]:
    df["Species"] = df["Species"].str.strip()

# Merge
merged_df = gi_df[["Species", "GI", "Order"]].merge(
    eq_df[["Species", "EQ Value"]].rename(columns={"EQ Value": "EQ"}),
    on="Species", how="inner"
).merge(
    fc_df[["Species", "Q_Score", "Num_Communities"]],
    on="Species", how="inner"
)

merged_df = merged_df.dropna(subset=["GI", "EQ", "Q_Score", "Num_Communities"])

# Standardize values
scaler = StandardScaler()
z_gi = scaler.fit_transform(merged_df[["GI"]])
z_eq = scaler.fit_transform(merged_df[["EQ"]])
z_q = scaler.fit_transform(merged_df[["Q_Score"]])
z_comm = scaler.fit_transform(merged_df[["Num_Communities"]])

# Define composite morphospace axes
merged_df["Z_struct"] = (z_gi + z_eq) / 2
merged_df["Z_func"] = (z_q + z_comm) / 2

# --- Plot Morphospace ---
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=merged_df,
    x="Z_struct", y="Z_func",
    hue="Order",
    palette="tab10", s=80, edgecolor="k"
)

# Add clade centroids
centroids = merged_df.groupby("Order")[["Z_struct", "Z_func"]].mean()
for order, row in centroids.iterrows():
    plt.scatter(row["Z_struct"], row["Z_func"], marker="X", s=200, label=f"{order} centroid")
    plt.text(row["Z_struct"], row["Z_func"], order, fontsize=9, weight="bold")

# Optional: confidence ellipses
def plot_ellipse(x, y, ax, n_std=2.0, **kwargs):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, theta, **kwargs)
    ax.add_patch(ellipse)

ax = plt.gca()
for order, group in merged_df.groupby("Order"):
    if len(group) > 5:  # only plot ellipses if group has enough samples
        plot_ellipse(group["Z_struct"], group["Z_func"], ax, alpha=0.15, color=ax._get_lines.get_next_color())

# Label outliers (e.g., >2 SD from mean in either axis)
zscore_struct = (merged_df["Z_struct"] - merged_df["Z_struct"].mean()) / merged_df["Z_struct"].std()
zscore_func = (merged_df["Z_func"] - merged_df["Z_func"].mean()) / merged_df["Z_func"].std()
outliers = merged_df[(abs(zscore_struct) > 2) | (abs(zscore_func) > 2)]
for _, row in outliers.iterrows():
    plt.annotate(row["Species"], (row["Z_struct"], row["Z_func"]), fontsize=8)

plt.xlabel("Structural Complexity (Z(GI, EQ))", fontsize=12)
plt.ylabel("Functional Complexity (Z(Modularity, Communities))", fontsize=12)
plt.title("Figure 3A: Morphospace of Structural vs Functional Complexity", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.close()

# --- PCA Validation (Supplement) ---
X = merged_df[["GI", "EQ", "Q_Score", "Num_Communities"]].values
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=merged_df["Order"], palette="tab10", s=80, edgecolor="k")
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
plt.title("Supplement: PCA Validation of Morphospace Dimensions", fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_pca_img, dpi=300)
plt.close()

# Print PCA loadings
loadings = pd.DataFrame(pca.components_.T, index=["GI", "EQ", "Q_Score", "Num_Communities"], columns=["PC1", "PC2"])
print("PCA Loadings:\n", loadings)
