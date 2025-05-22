import re
from pathlib import Path

def create_unique_species_list(input_path, output_path):
    """Process species list to remove numbered variants"""
    input_file = Path(input_path)
    output_file = Path(output_path)

    # Read original list and process names
    with open(input_file, 'r') as f:
        species = {re.sub(r'\d+$', '', line.strip()).capitalize() for line in f if line.strip()}

    # Write unique sorted list
    with open(output_file, 'w') as f:
        f.write('\n'.join(sorted(species)))

    print(f"Generated unique species list at: {output_file.resolve()}")

if __name__ == "__main__":
    # Define relative paths
    current_dir = Path(__file__).parent
    input_path = current_dir.parent / "data" / "species_list.txt"
    output_path = current_dir.parent / "data" / "unique_species_list.txt"

    create_unique_species_list(input_path, output_path)
