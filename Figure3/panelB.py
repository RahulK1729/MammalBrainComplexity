# MammalBrainComplexity\Figure3\panelB.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure3")
os.makedirs(fig_dir, exist_ok=True)
output_img = os.path.join(fig_dir, "figure3_panelB_clusters.png")
output_txt = os.path.join(fig_dir, "morphospace_clusters.txt")

# Load datasets
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))

# Clean species names
for df in [gi_df, eq_df, fc_df]:
    df["Species"] = df["Species"].str.strip().str.lower().str.replace(" ", "")

# Merge datasets
merged_df = gi_df[["Species", "GI"]].merge(
    eq_df[["Species", "EQ Value"]].rename(columns={"EQ Value": "EQ"}),
    on="Species", how="inner"
).merge(
    fc_df[["Species", "Q_Score", "Num_Communities"]],
    on="Species", how="inner"
)

# Drop missing
merged_df = merged_df.dropna(subset=["GI", "EQ", "Q_Score", "Num_Communities"])

# Z-score for axes
scaler = StandardScaler()
merged_df["Z_GI"] = scaler.fit_transform(merged_df[["GI"]])
merged_df["Z_EQ"] = scaler.fit_transform(merged_df[["EQ"]])
merged_df["Z_Q"] = scaler.fit_transform(merged_df[["Q_Score"]])
merged_df["Z_Comm"] = scaler.fit_transform(merged_df[["Num_Communities"]])

merged_df["Z_struct"] = (merged_df["Z_GI"] + merged_df["Z_EQ"]) / 2
merged_df["Z_func"] = (merged_df["Z_Q"] + merged_df["Z_Comm"]) / 2

# KMeans clustering (auto k = 2 as before)
kmeans = KMeans(n_clusters=2, random_state=42)
merged_df["Cluster"] = kmeans.fit_predict(merged_df[["Z_struct", "Z_func"]])

# Compute centroids for labeling top 3 closest species
centroids = merged_df.groupby("Cluster")[["Z_struct", "Z_func"]].mean().reset_index()

labels = []
for _, centroid in centroids.iterrows():
    cluster_num = centroid["Cluster"]
    sub = merged_df[merged_df["Cluster"] == cluster_num].copy()
    # Compute distance to centroid
    sub["dist_to_centroid"] = np.sqrt(
        (sub["Z_struct"] - centroid["Z_struct"])**2 + (sub["Z_func"] - centroid["Z_func"])**2
    )
    top3 = sub.nsmallest(3, "dist_to_centroid")
    labels.append(top3)

label_df = pd.concat(labels)


# Save cluster membership
with open(output_txt, "w") as f:
    for c in sorted(merged_df["Cluster"].unique()):
        f.write(f"Cluster {c}:\n")
        for s in merged_df[merged_df["Cluster"] == c]["Species"].tolist():
            f.write(f"{s}\n")
        f.write("\n")

# Plot
plt.figure(figsize=(10, 8))
palette = sns.color_palette("tab10", n_colors=merged_df["Cluster"].nunique())
sns.scatterplot(
    data=merged_df,
    x="Z_struct",
    y="Z_func",
    hue="Cluster",
    palette=palette,
    s=80,
    edgecolor="k"
)

# Annotate top 3 central species per cluster
for _, row in label_df.iterrows():
    plt.text(row["Z_struct"], row["Z_func"], row["Species"], fontsize=8, weight="bold")

plt.xlabel("Structural Complexity (Z-scored GI + EQ)")
plt.ylabel("Functional Complexity (Z-scored Modularity + Num Communities)")
plt.title("Figure 3B: Morphospace Clusters of Mammals")
plt.legend(title="Cluster", loc="best")
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel B saved to {output_img}")
print(f"📄 Cluster membership written to {output_txt}")
