import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import os

# Set file paths
base_dir = os.path.dirname(os.path.dirname(__file__))  # one level up from Figure3
gi_path = os.path.join(base_dir, 'data', 'gi_values.csv')
eq_path = os.path.join(base_dir, 'data', 'eq_values.csv')
fc_path = os.path.join(base_dir, 'data', 'functional_complexity_metrics.csv')
order_path = os.path.join(base_dir, 'data', 'species_order.csv')

# Load data
gi_df = pd.read_csv(gi_path)
eq_df = pd.read_csv(eq_path)
fc_df = pd.read_csv(fc_path)
order_df = pd.read_csv(order_path)

# Rename columns for consistency
eq_df = eq_df.rename(columns={'EQ Value': 'EQ'})
gi_df = gi_df.rename(columns={'Species': 'Species', 'GI': 'GI'})
fc_df = fc_df.rename(columns={'Species': 'Species'})
order_df = order_df.rename(columns={'Species': 'Species', 'Order': 'Order'})

# Clean species names
for df in [gi_df, eq_df, fc_df, order_df]:
    df['Species'] = df['Species'].str.strip().str.replace(" ", "").str.lower()

# Merge all dataframes
merged_df = eq_df[['Species', 'EQ']].merge(
    gi_df[['Species', 'GI']], on='Species', how='inner'
).merge(
    fc_df[['Species', 'Q_Score', 'Num_Communities']], on='Species', how='inner'
).merge(
    order_df[['Species', 'Order']], on='Species', how='left'
)

# Drop missing data
merged_df = merged_df.dropna(subset=['GI', 'EQ', 'Q_Score', 'Num_Communities'])

# Compute z-scores
merged_df['Structural_Complexity'] = zscore(merged_df['GI']) + zscore(merged_df['EQ'])
merged_df['Functional_Complexity'] = zscore(merged_df['Q_Score']) + zscore(merged_df['Num_Communities'])

# Plot
plt.figure(figsize=(10, 8))
sns.set(style='whitegrid', font_scale=1.2)

ax = sns.scatterplot(
    data=merged_df,
    x='Structural_Complexity',
    y='Functional_Complexity',
    hue='Order',
    palette='Set2',
    s=100,
    edgecolor='black'
)

# Optionally add species labels
# for _, row in merged_df.iterrows():
#     plt.text(row['Structural_Complexity'] + 0.05, row['Functional_Complexity'] + 0.05, row['Species'].capitalize(), fontsize=8)

plt.title('Figure 3A: Morphospace of Structural vs. Functional Complexity')
plt.xlabel('Structural Complexity (z(GI) + z(EQ))')
plt.ylabel('Functional Complexity (z(Modularity) + z(Num Communities))')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save figure
save_path = os.path.join(os.path.dirname(__file__), 'figure3_panelA_morphospace.png')
plt.savefig(save_path, dpi=300)
plt.show()
