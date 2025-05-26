import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths relative to current working directory (MammalBrainComplexity)
fc_path = os.path.join("data", "functional_complexity_metrics.csv")
order_path = os.path.join("data", "species_order.csv")

# Load and clean
fc_df = pd.read_csv(fc_path)
order_df = pd.read_csv(order_path)

fc_df['Species'] = fc_df['Species'].str.strip().str.replace(" ", "").str.lower()
order_df['Species'] = order_df['Species'].str.strip().str.replace(" ", "").str.lower()

df = pd.merge(fc_df[['Species', 'Q_Score']], order_df[['Species', 'Order']], on='Species')

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Order', y='Q_Score', data=df)
plt.xticks(rotation=45)
plt.ylabel("Modularity (Q Score)")
plt.title("Modularity Across Mammalian Orders")
plt.tight_layout()

# Save to Figure2/
output_path = os.path.join("Figure2", "modularity_boxplot.png")
plt.savefig(output_path, dpi=300)
plt.show()
