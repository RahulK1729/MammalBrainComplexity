import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import KBinsDiscretizer

# -------------------------
# Paths
# -------------------------
base_dir = os.path.dirname(os.path.dirname(__file__))  # MammalBrainComplexity
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure4")
os.makedirs(fig_dir, exist_ok=True)

fc_file = os.path.join(data_dir, "functional_complexity_metrics.csv")
soc_file = os.path.join(data_dir, "sociality_scores.csv")

output_img = os.path.join(fig_dir, "figure4_panelC_modularity_sociality.png")
output_txt = os.path.join(fig_dir, "sociality_modularity_extremes.txt")

# -------------------------
# Load Data
# -------------------------
fc_df = pd.read_csv(fc_file)
soc_df = pd.read_csv(soc_file)

# Clean species names
for df in [fc_df, soc_df]:
    df["Species"] = df["Species"].str.strip()

# Merge
merged_df = pd.merge(fc_df, soc_df, on="Species", how="inner")
merged_df = merged_df.dropna(subset=["Q_Score", "Sociality_Score"])

# -------------------------
# Bin sociality scores into 3 categories
# -------------------------
est = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")
merged_df["Sociality_Group"] = est.fit_transform(merged_df[["Sociality_Score"]]).astype(int)

group_labels = {0: "Solitary", 1: "Mixed", 2: "Social"}
merged_df["Sociality_Group_Label"] = merged_df["Sociality_Group"].map(group_labels)

# -------------------------
# Find extremes per group
# -------------------------
extremes = []
for grp, label in group_labels.items():
    grp_df = merged_df[merged_df["Sociality_Group"] == grp]
    if not grp_df.empty:
        max_row = grp_df.loc[grp_df["Q_Score"].idxmax()]
        min_row = grp_df.loc[grp_df["Q_Score"].idxmin()]
        extremes.append(max_row)
        extremes.append(min_row)

extremes_df = pd.DataFrame(extremes)

# Save extremes to text
with open(output_txt, "w") as f:
    for grp, label in group_labels.items():
        f.write(f"{label} group:\n")
        grp_ext = extremes_df[extremes_df["Sociality_Group"] == grp]
        for _, row in grp_ext.iterrows():
            f.write(f"- {row['Species']}: Q_Score={row['Q_Score']}, Sociality={row['Sociality_Score']}\n")
        f.write("\n")

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x="Sociality_Group_Label", y="Q_Score", data=merged_df, palette="coolwarm")
sns.stripplot(x="Sociality_Group_Label", y="Q_Score", data=merged_df,
              color="black", size=5, jitter=True, alpha=0.6)

# Annotate extremes
for _, row in extremes_df.iterrows():
    plt.annotate(row["Species"], 
                 xy=(row["Sociality_Group_Label"], row["Q_Score"]),
                 xytext=(5, 3),
                 textcoords="offset points",
                 fontsize=8,
                 color="black")

plt.xlabel("Sociality Group", fontsize=12)
plt.ylabel("Functional Network Modularity (Q_Score)", fontsize=12)
plt.title("Figure 4C: Modularity by Sociality", fontsize=14)

plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel C saved to {output_img}")
print(f"📄 Extremes saved to {output_txt}")
