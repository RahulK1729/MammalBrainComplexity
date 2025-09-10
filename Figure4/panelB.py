# Figure 4 Panel B: Efficiency vs Modularity with efficiency/modularity extremes

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure4")
os.makedirs(fig_dir, exist_ok=True)

output_img = os.path.join(fig_dir, "figure4_panelB_efficiency_vs_modularity_ratio.png")
output_txt = os.path.join(fig_dir, "figure4_panelB_ratio_extremes.txt")

# Load data
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))

# Drop missing values
df = fc_df.dropna(subset=["Q_Score", "Num_Communities"])

# Compute proxy for efficiency
df["Efficiency"] = 1 / (1 + df["Num_Communities"])

# Compute efficiency/modularity ratio
df["Efficiency_Modularity_Ratio"] = df["Efficiency"] / df["Q_Score"]

# Identify top/bottom species by ratio
top_ratio = df.nlargest(2, "Efficiency_Modularity_Ratio")
bottom_ratio = df.nsmallest(2, "Efficiency_Modularity_Ratio")
extremes = pd.concat([top_ratio, bottom_ratio])

# Save extremes to text file
with open(output_txt, "w") as f:
    f.write("Top 2 Efficiency/Modularity Ratio:\n")
    for _, row in top_ratio.iterrows():
        f.write(f"- {row['Species']}: {row['Efficiency_Modularity_Ratio']:.3f}\n")
    f.write("\nBottom 2 Efficiency/Modularity Ratio:\n")
    for _, row in bottom_ratio.iterrows():
        f.write(f"- {row['Species']}: {row['Efficiency_Modularity_Ratio']:.3f}\n")

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Q_Score",
    y="Efficiency",
    color="skyblue",
    s=70,
    edgecolor="black"
)

# Regression line
sns.regplot(
    data=df,
    x="Q_Score",
    y="Efficiency",
    scatter=False,
    color="black",
    line_kws={"linestyle": "--"}
)

# Annotate extremes
for _, row in extremes.iterrows():
    plt.annotate(
        row["Species"],
        (row["Q_Score"], row["Efficiency"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=9,
        color='red',
        arrowprops=dict(arrowstyle="->", color='red', lw=1)
    )

plt.xlabel("Modularity (Q Score)", fontsize=12)
plt.ylabel("Network Efficiency (1 / (1 + Communities))", fontsize=12)
plt.title("Figure 4B: Efficiency vs Modularity (Efficiency/Modularity Ratio)", fontsize=14)

plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel B saved to {output_img}")
print(f"📄 Efficiency/Modularity ratio extremes written to {output_txt}")
