# MammalBrainComplexity
Analysis of brain complexity in mammals using the MaMI dataset in the context of structural and functional complexity. Includes encephalization quotient (EQ), gyrification index (GI), and functional network properties (modularity, community structure, efficiency)

# Setup
# On Windows
python -m venv ../MaMI_complexity_venv

# On macOS/Linux
python3 -m venv ../MaMI_complexity_venv

# On Windows
.\MaMI_complexity_venv\Scripts\activate

# On macOS/Linux
source ../MaMI_complexity_venv/bin/activate

# Install Dependencies
pip install -r requirements.txt

# If adding necessary libraries, please use
pip freeze > requirements.txt
