# Figure 4 Panel A: Schematic functional networks (low, medium, high complexity)

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
fig_dir = os.path.join(base_dir, "Figure4")
os.makedirs(fig_dir, exist_ok=True)
output_img = os.path.join(fig_dir, "figure4_panelA_networks.png")

# Load functional metrics
fc_df = pd.read_csv(os.path.join(data_dir, "functional_complexity_metrics.csv"))

# Drop NAs
fc_df = fc_df.dropna(subset=["Q_Score", "Num_Communities"])

# Rank by complexity (Q_Score + communities)
fc_df["Complexity"] = (
    (fc_df["Q_Score"] - fc_df["Q_Score"].min()) / (fc_df["Q_Score"].max() - fc_df["Q_Score"].min())
    + (fc_df["Num_Communities"] - fc_df["Num_Communities"].min()) / (fc_df["Num_Communities"].max() - fc_df["Num_Communities"].min())
)

# Pick examples: 2 low, 2 mid, 2 high
low_species = fc_df.nsmallest(2, "Complexity")
mid_species = fc_df.iloc[fc_df["Complexity"].sort_values().index[len(fc_df)//2 - 1 : len(fc_df)//2 + 1]]
high_species = fc_df.nlargest(2, "Complexity")

examples = pd.concat([low_species, mid_species, high_species])

# Function to generate schematic network
def generate_network(num_communities, q_score, seed=42):
    np.random.seed(seed)
    G = nx.Graph()

    # Number of nodes ~ communities * scaling factor
    num_nodes = int(num_communities * 6) if num_communities > 0 else 6
    nodes_per_comm = max(3, num_nodes // num_communities) if num_communities > 0 else num_nodes

    # Build communities
    community_nodes = []
    node_id = 0
    for c in range(num_communities):
        comm_nodes = [node_id + i for i in range(nodes_per_comm)]
        node_id += nodes_per_comm
        community_nodes.append(comm_nodes)
        # Add edges within communities (dense)
        for i in comm_nodes:
            for j in comm_nodes:
                if i < j and np.random.rand() < 0.6 + 0.3 * q_score:  # more edges with higher modularity
                    G.add_edge(i, j)

    # Add sparse inter-community edges
    for i in range(len(community_nodes)):
        for j in range(i + 1, len(community_nodes)):
            if np.random.rand() < 0.1:  # sparse
                n1 = np.random.choice(community_nodes[i])
                n2 = np.random.choice(community_nodes[j])
                G.add_edge(n1, n2)

    return G

# Plot networks
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (_, row) in zip(axes, examples.iterrows()):
    species = row["Species"]
    q = row["Q_Score"]
    comms = int(round(row["Num_Communities"]))

    G = generate_network(comms if comms > 0 else 2, q)

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        with_labels=False,
        node_size=80,
        node_color="red",
        edge_color="gray",
        width=0.7
    )
    ax.set_title(f"{species}\nQ={q:.2f}, Comms={comms}", fontsize=11)
    ax.axis("off")

plt.suptitle("Figure 4A: Schematic Functional Networks\n(Low, Medium, High Complexity)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_img, dpi=300)
plt.show()
plt.close()

print(f"✅ Panel A saved to {output_img}")