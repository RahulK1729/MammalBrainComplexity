import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure5")
os.makedirs(fig_dir, exist_ok=True)

output_img = os.path.join(fig_dir, "figure5_panelC_deviation_by_sociality.png")
output_txt = os.path.join(fig_dir, "sociality_extremes.txt")

# Load data
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))
soc_df = pd.read_csv(os.path.join(data_dir, "sociality_scores.csv"))

# Clean names
for df in [gi_df, eq_df, fc_df, soc_df]:
    df["Species"] = df["Species"].str.strip()

# Merge datasets
merged_df = gi_df[["Species", "GI"]].merge(
    eq_df[["Species", "EQ Value"]].rename(columns={"EQ Value": "EQ"}), on="Species", how="inner"
).merge(
    fc_df[["Species", "Q_Score", "Num_Communities"]], on="Species", how="inner"
).merge(
    soc_df[["Species", "Sociality_Score"]], on="Species", how="inner"
).dropna()

# Standardize
scaler = StandardScaler()
merged_df["GI_z"] = scaler.fit_transform(merged_df[["GI"]])
merged_df["EQ_z"] = scaler.fit_transform(merged_df[["EQ"]])
merged_df["Q_z"] = scaler.fit_transform(merged_df[["Q_Score"]])
merged_df["C_z"] = scaler.fit_transform(merged_df[["Num_Communities"]])

# Define morphospace axes
merged_df["Structural"] = (merged_df["GI_z"] + merged_df["EQ_z"]) / 2
merged_df["Functional"] = (merged_df["Q_z"] + merged_df["C_z"]) / 2

# Regression for expected functional complexity
X = sm.add_constant(merged_df["Structural"])
y = merged_df["Functional"]
model = sm.OLS(y, X).fit()
merged_df["Predicted"] = model.predict(X)
merged_df["Deviation"] = merged_df["Functional"] - merged_df["Predicted"]

# Bin sociality into categories
def categorize_sociality(score):
    if score <= 3:
        return "Solitary"
    elif score <= 6:
        return "Mixed"
    else:
        return "Social"

merged_df["Sociality_Category"] = merged_df["Sociality_Score"].apply(categorize_sociality)

# Find extremes per category
extremes = []
for cat in ["Solitary", "Mixed", "Social"]:
    sub = merged_df[merged_df["Sociality_Category"] == cat]
    if len(sub) > 0:
        max_dev = sub.loc[sub["Deviation"].idxmax()]
        min_dev = sub.loc[sub["Deviation"].idxmin()]
        extremes.extend([max_dev, min_dev])
extremes_df = pd.DataFrame(extremes)

# Save extremes to text
with open(output_txt, "w") as f:
    f.write("Most extreme deviations by sociality category:\n\n")
    for cat in ["Solitary", "Mixed", "Social"]:
        sub = extremes_df[extremes_df["Sociality_Category"] == cat]
        for _, row in sub.iterrows():
            f.write(f"{cat}: {row['Species']} deviation = {row['Deviation']:.3f}\n")

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x="Sociality_Category", y="Deviation", data=merged_df,
            order=["Solitary", "Mixed", "Social"], palette="coolwarm")
sns.stripplot(x="Sociality_Category", y="Deviation", data=merged_df,
              order=["Solitary", "Mixed", "Social"], color="black", size=4, alpha=0.7)

plt.axhline(0, color="gray", linestyle="--", lw=1)

# Annotate extremes
for _, row in extremes_df.iterrows():
    plt.annotate(
        row["Species"],
        (["Solitary", "Mixed", "Social"].index(row["Sociality_Category"]), row["Deviation"]),
        textcoords="offset points",
        xytext=(5,5),
        ha='left',
        fontsize=9,
        color="black",
    )

plt.title("Figure 5C: Deviation by Sociality Category", fontsize=14)
plt.ylabel("Deviation (Functional - Predicted)")
plt.xlabel("Sociality Category")
plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()

print(f"✅ Panel C saved to {output_img}")
print(f"📄 Extremes written to {output_txt}")
