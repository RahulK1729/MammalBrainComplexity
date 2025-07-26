import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore
from scipy.spatial import ConvexHull
import numpy as np
import os

# Set file paths
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from Figure3
gi_path = os.path.join(base_dir, 'data', 'gi_values.csv')
eq_path = os.path.join(base_dir, 'data', 'eq_values.csv')
fc_path = os.path.join(base_dir, 'data', 'functional_complexity_metrics.csv')
order_path = os.path.join(base_dir, 'data', 'species_order.csv')

# Load data
gi_df = pd.read_csv(gi_path)
eq_df = pd.read_csv(eq_path)
fc_df = pd.read_csv(fc_path)
order_df = pd.read_csv(order_path)

# Clean and rename
eq_df = eq_df.rename(columns={'EQ Value': 'EQ'})
gi_df = gi_df.rename(columns={'GI': 'GI'})
for df in [gi_df, eq_df, fc_df, order_df]:
    df['Species'] = df['Species'].str.strip().str.replace(" ", "").str.lower()

# Merge
merged_df = eq_df[['Species', 'EQ']].merge(
    gi_df[['Species', 'GI']], on='Species', how='inner'
).merge(
    fc_df[['Species', 'Q_Score', 'Num_Communities']], on='Species', how='inner'
).merge(
    order_df[['Species', 'Order']], on='Species', how='left'
)

merged_df = merged_df.dropna(subset=['GI', 'EQ', 'Q_Score', 'Num_Communities'])

# Calculate z-scores
merged_df['Structural_Complexity'] = zscore(merged_df['GI']) + zscore(merged_df['EQ'])
merged_df['Functional_Complexity'] = zscore(merged_df['Q_Score']) + zscore(merged_df['Num_Communities'])

# Prepare features for clustering
X = merged_df[['Structural_Complexity', 'Functional_Complexity']].values

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(X)

# Plotting
plt.figure(figsize=(10, 8))
sns.set(style='whitegrid', font_scale=1.2)
palette = sns.color_palette('Set1', n_colors=3)

# Scatter plot with clusters
sns.scatterplot(
    data=merged_df,
    x='Structural_Complexity',
    y='Functional_Complexity',
    hue='Cluster',
    palette=palette,
    s=100,
    edgecolor='black',
    legend='full'
)

# Convex hulls
for cluster_id in merged_df['Cluster'].unique():
    cluster_points = merged_df[merged_df['Cluster'] == cluster_id][['Structural_Complexity', 'Functional_Complexity']].values
    if len(cluster_points) >= 3:
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.2, color=palette[cluster_id], linewidth=0)

plt.title('Figure 3B: Clustering in Morphospace')
plt.xlabel('Structural Complexity (z(GI) + z(EQ))')
plt.ylabel('Functional Complexity (z(Modularity) + z(Num Communities))')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save figure
save_path = os.path.join(os.path.dirname(__file__), 'figure3_panelB_clusters.png')
plt.savefig(save_path, dpi=300)
plt.show()

# Generate cluster listings
cluster_summary = ""

for cluster_id in sorted(merged_df['Cluster'].unique()):
    species_list = merged_df[merged_df['Cluster'] == cluster_id]['Species'].tolist()
    cluster_summary += f"Cluster {cluster_id} ({len(species_list)} species):\n"
    cluster_summary += "\n".join(f"  - {species}" for species in species_list)
    cluster_summary += "\n\n"

# Save to text file
output_txt_path = os.path.join(os.path.dirname(__file__), 'figure3_cluster_membership.txt')
with open(output_txt_path, 'w') as f:
    f.write(cluster_summary)

print(f"Cluster memberships saved to: {output_txt_path}")

