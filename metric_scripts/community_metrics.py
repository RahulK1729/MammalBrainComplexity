import numpy as np
import csv
from pathlib import Path

# Get the directory where the current script is located
script_dir = Path(__file__).parent

# Path to the species list (relative to the script directory)
species_list_file = script_dir.parent / "data" / "species_list.txt"

# Path for the output CSV file (relative to the script directory)
output_csv_file = script_dir.parent / "data" / "community_metrics.csv"

# Function to calculate Q-score from modularity scores (average)
def calculate_q_score(modularity_scores):
    # Calculate the maximum of the modularity scores
    return np.max(modularity_scores)

# Function to calculate number of communities from the communities file
def calculate_num_communities(communities):
    return len(np.unique(communities))

# Read species names from the file
species_list = []
with open(species_list_file, 'r') as f:
    species_list = [line.strip() for line in f.readlines()]

# Open the output CSV file in write mode
with open(output_csv_file, mode='w', newline='') as csvfile:
    fieldnames = ['Species', 'Q_Score', 'Num_Communities']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    # Iterate through species and calculate metrics
    for species in species_list:
        # Relative paths to the files for each species in the 'raw_results' folder
        community_file = script_dir.parent.parent / "raw_results" / "res_100" / "rich_club" / f"{species}_communities.npy"
        modularity_file = script_dir.parent.parent / "raw_results" / "res_100" / "rich_club" / f"{species}_modularity_scores.npy"

        # Check if the files exist
        if community_file.exists() and modularity_file.exists():
            # Load the communities and modularity scores
            communities = np.load(community_file)
            modularity_scores = np.load(modularity_file)

            # Calculate Q score and number of communities
            q_score = calculate_q_score(modularity_scores)
            num_communities = calculate_num_communities(communities)

            # Write the results to the CSV
            writer.writerow({'Species': species, 'Q_Score': q_score, 'Num_Communities': num_communities})
        else:
            print(f"Missing data for {species}, skipping...")

print("Finished writing metrics to community_metrics.csv")