import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")

# Modify these paths as needed
data_folder = Path(__file__).resolve().parents[1] / "data"
output_folder = Path(__file__).resolve().parent

eq_file = data_folder / "eq_values.csv"
gi_file = data_folder / "gi_values.csv"

# Load data
eq_df = pd.read_csv(eq_file)
gi_df = pd.read_csv(gi_file)

# Merge on Species, keep relevant columns
df = pd.merge(eq_df[['Species', 'EQ Value']], gi_df[['Species', 'Order', 'GI']], on='Species')

# Drop missing and 'Other' category
df.dropna(subset=["EQ Value", "GI", "Order"], inplace=True)
df = df[df['Order'].str.lower() != 'other']

# Order by frequency descending
order_list = df['Order'].value_counts().sort_values(ascending=False).index.tolist()
palette = sns.color_palette("Set2", len(order_list))
order_palette = dict(zip(order_list, palette))

# === Panel C EQ plot ===
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x="Order", y="EQ Value", data=df, ax=ax,
    hue="Order", palette=order_palette,
    dodge=False, legend=False, order=order_list
)

sns.stripplot(
    x="Order", y="EQ Value", data=df, ax=ax,
    hue="Order", palette=order_palette,
    dodge=True, legend=False,
    size=4, jitter=True, order=order_list
)

ax.set_title("Encephalization Quotient (EQ) by Mammalian Order", pad=20)
ax.set_xlabel("Order")
ax.set_ylabel("EQ Value")
ax.tick_params(axis='x', rotation=45)

# Annotate counts
for i, order in enumerate(order_list):
    count = df[df["Order"] == order].shape[0]
    y_pos = df["EQ Value"].max() * 1.05
    ax.text(i, y_pos, f'n={count}', ha='center', fontsize=10)

plt.tight_layout()
eq_output = output_folder / "Figure1_PanelC_EQ.png"
plt.savefig(eq_output, dpi=300)
plt.show()
print(f"Saved EQ plot to: {eq_output}")

# === Panel C GI plot ===
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    x="Order", y="GI", data=df, ax=ax,
    hue="Order", palette=order_palette,
    dodge=False, legend=False, order=order_list
)

sns.stripplot(
    x="Order", y="GI", data=df, ax=ax,
    hue="Order", palette=order_palette,
    dodge=True, legend=False,
    size=4, jitter=True, order=order_list
)

ax.set_title("Gyrification Index (GI) by Mammalian Order", pad=20)
ax.set_xlabel("Order")
ax.set_ylabel("GI")
ax.tick_params(axis='x', rotation=45)

for i, order in enumerate(order_list):
    count = df[df["Order"] == order].shape[0]
    y_pos = df["GI"].max() * 1.05
    ax.text(i, y_pos, f'n={count}', ha='center', fontsize=10)

plt.tight_layout()
gi_output = output_folder / "Figure1_PanelC_GI.png"
plt.savefig(gi_output, dpi=300)
plt.show()
print(f"Saved GI plot to: {gi_output}")
