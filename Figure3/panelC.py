# MammalBrainComplexity\Figure3\panelC.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure3")
os.makedirs(fig_dir, exist_ok=True)
output_img = os.path.join(fig_dir, "figure3_panelC_sociality_gradient.png")
output_txt = os.path.join(fig_dir, "sociality_extremes.txt")

# Load datasets
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))
soc_df = pd.read_csv(os.path.join(data_dir, "sociality_scores.csv"))

# Clean species names
for df in [gi_df, eq_df, fc_df, soc_df]:
    df["Species"] = df["Species"].str.strip().str.lower().str.replace(" ", "")

# Merge datasets
merged_df = gi_df[["Species", "GI"]].merge(
    eq_df[["Species", "EQ Value"]].rename(columns={"EQ Value": "EQ"}),
    on="Species", how="inner"
).merge(
    fc_df[["Species", "Q_Score", "Num_Communities"]],
    on="Species", how="inner"
).merge(
    soc_df[["Species", "Sociality_Score"]],
    on="Species", how="inner"
)

# Drop missing
merged_df = merged_df.dropna(subset=["GI", "EQ", "Q_Score", "Num_Communities", "Sociality_Score"])

# Compute Z-axes
scaler = StandardScaler()
merged_df["Z_struct"] = (scaler.fit_transform(merged_df[["GI"]]) + scaler.fit_transform(merged_df[["EQ"]])) / 2
merged_df["Z_func"] = (scaler.fit_transform(merged_df[["Q_Score"]]) + scaler.fit_transform(merged_df[["Num_Communities"]])) / 2

# Normalize sociality score for color
color_scaler = MinMaxScaler()
merged_df["sociality_norm"] = color_scaler.fit_transform(merged_df[["Sociality_Score"]])

# Identify 3 most social and 3 most solitary
most_social = merged_df.nlargest(3, "Sociality_Score")
least_social = merged_df.nsmallest(3, "Sociality_Score")
extremes = pd.concat([most_social, least_social])

# Save to text file
with open(output_txt, "w") as f:
    f.write("Most Social Species:\n")
    for _, row in most_social.iterrows():
        f.write(f"- {row['Species']}: {row['Sociality_Score']}\n")
    f.write("\nMost Solitary Species:\n")
    for _, row in least_social.iterrows():
        f.write(f"- {row['Species']}: {row['Sociality_Score']}\n")

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    merged_df["Z_struct"],
    merged_df["Z_func"],
    c=merged_df["sociality_norm"],
    cmap="coolwarm",
    s=80,
    edgecolor="k"
)

# Labels and colorbar
plt.xlabel("Structural Complexity (Z-scored GI + EQ)", fontsize=12)
plt.ylabel("Functional Complexity (Z-scored Modularity + Num Communities)", fontsize=12)
plt.title("Figure 3C: Sociality Gradient Across Morphospace", fontsize=14)

# Colorbar
cbar = plt.colorbar(scatter, label="Sociality Score (0=Solitary, 1=Social)")
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["Solitary", "Mixed", "Social"])

# Annotate extremes
for _, row in extremes.iterrows():
    plt.annotate(
        row["Species"],
        (row["Z_struct"], row["Z_func"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=9,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=1)
    )

# Save
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel C saved to {output_img}")
print(f"📄 Sociality extremes written to {output_txt}")
