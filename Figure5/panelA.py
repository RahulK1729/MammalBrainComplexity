# Figure 5 Panel A: Outliers in Morphospace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure5")
os.makedirs(fig_dir, exist_ok=True)
output_img = os.path.join(fig_dir, "figure5_panelA_outliers.png")
output_txt = os.path.join(fig_dir, "outliers_panelA.txt")

# Load datasets
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))
soc_df = pd.read_csv(os.path.join(data_dir, "sociality_scores.csv"))

# Clean species names
for df in [gi_df, eq_df, fc_df, soc_df]:
    df["Species"] = df["Species"].str.strip()

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
merged_df = merged_df.dropna(subset=["GI", "EQ", "Q_Score", "Num_Communities"])

# Compute structural & functional complexity (Z-scores averaged)
scaler = StandardScaler()
merged_df["GI_z"] = scaler.fit_transform(merged_df[["GI"]])
merged_df["EQ_z"] = scaler.fit_transform(merged_df[["EQ"]])
merged_df["Q_z"] = scaler.fit_transform(merged_df[["Q_Score"]])
merged_df["Comm_z"] = scaler.fit_transform(merged_df[["Num_Communities"]])

merged_df["Structural"] = (merged_df["GI_z"] + merged_df["EQ_z"]) / 2
merged_df["Functional"] = (merged_df["Q_z"] + merged_df["Comm_z"]) / 2

# Fit regression Functional ~ Structural
X = merged_df[["Structural"]].values
y = merged_df["Functional"].values
model = LinearRegression()
model.fit(X, y)
merged_df["Predicted_Functional"] = model.predict(X)
merged_df["Residual"] = merged_df["Functional"] - merged_df["Predicted_Functional"]

# Identify outliers
top_outliers = merged_df.nlargest(3, "Residual")
bottom_outliers = merged_df.nsmallest(3, "Residual")
outliers = pd.concat([top_outliers, bottom_outliers])

# Save outlier info to text file
with open(output_txt, "w") as f:
    f.write("Top Positive Outliers:\n")
    for _, row in top_outliers.iterrows():
        f.write(f"- {row['Species']}: Residual = {row['Residual']:.3f}\n")
    f.write("\nTop Negative Outliers:\n")
    for _, row in bottom_outliers.iterrows():
        f.write(f"- {row['Species']}: Residual = {row['Residual']:.3f}\n")

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=merged_df,
    x="Structural",
    y="Functional",
    color="gray",
    alpha=0.7,
    edgecolor="k"
)

# Regression line
x_vals = np.linspace(merged_df["Structural"].min(), merged_df["Structural"].max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
plt.plot(x_vals, y_vals, color="black", linestyle="--", label="Regression line")

# Highlight outliers
sns.scatterplot(
    data=outliers,
    x="Structural",
    y="Functional",
    color="red",
    s=120,
    edgecolor="k",
    label="Outliers"
)

# Annotate outliers
for _, row in outliers.iterrows():
    plt.annotate(
        row["Species"],
        (row["Structural"], row["Functional"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", lw=1, color="red")
    )

plt.xlabel("Structural Complexity (Z(GI) + Z(EQ)) / 2", fontsize=12)
plt.ylabel("Functional Complexity (Z(Modularity) + Z(Communities)) / 2", fontsize=12)
plt.title("Figure 5A: Outliers in Brain Complexity Morphospace", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel A saved to {output_img}")
print(f"📄 Outliers written to {output_txt}")
