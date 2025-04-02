import os

# Path to the 'raw_results' folder (assuming both are at the same level)
raw_results_path = '../../raw_results/res_100'  # raw_results folder at the same level as repo

# Output file to save the folder structure
# modify as needed
output_file = 'res_100_structure.txt'

# Check if the raw_results folder exists
if os.path.exists(raw_results_path) and os.path.isdir(raw_results_path):
    print(f"Successfully found 'raw_results' directory at: {raw_results_path}")
    
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Walk through the 'raw_results' folder and write the folder structure to the file
        for root, dirs, files in os.walk(raw_results_path):
            # Write the current directory path
            f.write(f"{root}\n")
            
            # Write the filenames in the current directory, if any
            if files:
                for file in files:
                    f.write(f"    {file}\n")
            else:
                f.write("    No files in this folder\n")
    
    print(f"Folder structure has been written to {output_file}")
else:
    print(f"Could not find the 'raw_results' directory. Please check the path.")
