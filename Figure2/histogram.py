import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import linregress

# Adjusted base_dir and path
base_dir = os.path.abspath(os.path.join(".."))  # one level up: MammalBrainComplexity
raw_results_dir = os.path.join(base_dir, "raw_results", "res_100", "community_detection")
output_dir = os.path.join("..", "Figure2")  # relative path from MammalBrainComplexity folder

species_files = [f for f in os.listdir(raw_results_dir) if f.endswith("_communities.npy")]

all_community_sizes = []

for filename in species_files:
    file_path = os.path.join(raw_results_dir, filename)
    try:
        data = np.load(file_path, allow_pickle=True)
        # Assuming data is a list/array of community assignments per node
        # Community sizes: count number of nodes per unique community label
        unique, counts = np.unique(data, return_counts=True)
        community_sizes = counts
        all_community_sizes.extend(community_sizes)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")

# Check minimum community size
if all_community_sizes:
    print("Minimum community size found:", min(all_community_sizes))
else:
    print("No community sizes loaded!")

# Count frequencies
size_counts = Counter(all_community_sizes)
sizes = np.array(sorted(size_counts.keys()))
frequencies = np.array([size_counts[size] for size in sizes])

# Plot histogram
plt.figure(figsize=(8, 6))
plt.bar(sizes, frequencies, color='skyblue', edgecolor='black', alpha=0.7)

plt.xlabel('Community Size (Number of Nodes)')
plt.ylabel('Frequency')
plt.title('Distribution of Community Sizes Across Mammals')

# Fit log-log regression if possible
if np.all(sizes > 0) and np.all(frequencies > 0):
    log_sizes = np.log10(sizes)
    log_freq = np.log10(frequencies)
    slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_freq)
    print(f"Log-Log regression: slope={slope:.3f}, intercept={intercept:.3f}, R^2={r_value**2:.3f}")

    fit_line = 10**(intercept + slope * log_sizes)
    plt.plot(sizes, fit_line, color='red', lw=2, label='Log-Log Fit')

plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save figure
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "panel_d_community_size_histogram.png"), dpi=300)

plt.show()
