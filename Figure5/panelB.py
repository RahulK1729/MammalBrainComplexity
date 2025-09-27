# panelB.py - Figure 5 Panel B: Deviations by Order Groupings

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# === Paths ===
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure5")
os.makedirs(fig_dir, exist_ok=True)

output_img = os.path.join(fig_dir, "figure5_panelB_deviation_by_order.png")
output_txt = os.path.join(fig_dir, "orders_grouping.txt")

# === Load data ===
gi_df = pd.read_csv(os.path.join(data_dir, "gi_values.csv"))
eq_df = pd.read_csv(os.path.join(data_dir, "eq_values.csv"))
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))
orders_df = pd.read_csv(os.path.join(data_dir, "species_order.csv"))

# Clean species names
for df in [gi_df, eq_df, fc_df, orders_df]:
    df["Species"] = df["Species"].str.strip()

# Merge into one dataframe
merged_df = gi_df[["Species", "GI"]].merge(
    eq_df[["Species", "EQ Value"]].rename(columns={"EQ Value": "EQ"}),
    on="Species", how="inner"
).merge(
    fc_df[["Species", "Q_Score", "Num_Communities"]],
    on="Species", how="inner"
).merge(
    orders_df[["Species", "Order"]],
    on="Species", how="inner"
)

# Drop missing
merged_df = merged_df.dropna(subset=["GI", "EQ", "Q_Score", "Num_Communities", "Order"])

# === Compute morphospace axes ===
merged_df["Structural"] = (merged_df["GI"] - merged_df["GI"].mean()) / merged_df["GI"].std() \
                        + (merged_df["EQ"] - merged_df["EQ"].mean()) / merged_df["EQ"].std()
merged_df["Functional"] = (merged_df["Q_Score"] - merged_df["Q_Score"].mean()) / merged_df["Q_Score"].std() \
                        + (merged_df["Num_Communities"] - merged_df["Num_Communities"].mean()) / merged_df["Num_Communities"].std()

# === Fit regression to compute deviation ===
X = merged_df[["Structural"]].values
y = merged_df["Functional"].values
model = LinearRegression().fit(X, y)
merged_df["Predicted_Functional"] = model.predict(X)
merged_df["Deviation"] = merged_df["Functional"] - merged_df["Predicted_Functional"]

# === Collapse orders into "big 5" + Other ===
big5 = ["Primates", "Carnivora", "Artiodactyla", "Chiroptera", "Rodentia"]

def assign_group(order):
    return order if order in big5 else "Other"

merged_df["Order_Group"] = merged_df["Order"].apply(assign_group)

# === Compute average deviation per group ===
grouped = merged_df.groupby("Order_Group")["Deviation"].mean().reset_index()

# === Save list of what is in "Other" ===
other_orders = merged_df.loc[~merged_df["Order"].isin(big5), "Order"].unique()

with open(output_txt, "w") as f:
    f.write("Big 5 Orders (kept separate):\n")
    for b in big5:
        f.write(f"- {b}\n")
    f.write("\nOther Orders (collapsed into 'Other'):\n")
    for o in sorted(other_orders):
        f.write(f"- {o}\n")

# === Plot ===
plt.figure(figsize=(8, 6))
sns.barplot(data=grouped, x="Order_Group", y="Deviation", palette="muted", edgecolor="k")

plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Order Group", fontsize=12)
plt.ylabel("Mean Deviation (Functional - Predicted)", fontsize=12)
plt.title("Figure 5B: Deviation from Morphospace Regression by Order Group", fontsize=14)

plt.tight_layout()
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel B saved to {output_img}")
print(f"📄 Orders grouping written to {output_txt}")
