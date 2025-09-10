import os

# Directory where crawler.py lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Repo root (two levels up from crawler.py)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Path to raw_results\res_100
raw_results_path = os.path.join(project_root, "raw_results", "res_100")

# Output file in the same folder as crawler.py
output_file = os.path.join(script_dir, "res_100_structure_1.txt")

# Check if the raw_results folder exists
if os.path.exists(raw_results_path) and os.path.isdir(raw_results_path):
    print(f"Successfully found 'raw_results' directory at: {raw_results_path}")
    
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Walk through the 'raw_results' folder and write the folder structure to the file
        for root, dirs, files in os.walk(raw_results_path):
            # Write the current directory path
            f.write(f"{root}\n")
            
            # # Write the filenames in the current directory, if any
            # if files:
            #     for file in files:
            #         f.write(f"    {file}\n")
            # else:
            #     f.write("    No files in this folder\n")
            if dirs:
                for dir in dirs:
                    f.write(f"  {dir}\n")
            else:
                f.write("   No dirs in tihs folder\n")
    
    print(f"Folder structure has been written to {output_file}")
else:
    print(f"Could not find the 'raw_results' directory. Please check the path.")
