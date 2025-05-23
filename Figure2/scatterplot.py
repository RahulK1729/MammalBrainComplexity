import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress, pearsonr, mode, spearmanr, kendalltau
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
import statsmodels.api as sm

# === Paths ===
root = Path(__file__).resolve().parent.parent
data_path = root / "data" / "eq_values.csv"
npy_dir = root.parent / "raw_results" / "res_100" / "community_detection"
output_path = Path(__file__).resolve().parent / "Figure2_PanelB_Communities_vs_BrainMass.png"
diagnostic_path = Path(__file__).resolve().parent / "diagnostic_plots.png"

# === Load EQ Data ===
eq_df = pd.read_csv(data_path)
eq_df.dropna(subset=["Species", "Brain Mass (g)"], inplace=True)

# === Helper Function: Find First Matching File ===
def find_species_file(species, directory):
    """Check for files like {species}_communities.npy or {species}1_communities.npy."""
    # Try exact match first
    exact_path = directory / f"{species}_communities.npy"
    if exact_path.exists():
        return exact_path
    
    # Try variants with numbers (e.g., Agouti1_communities.npy)
    for candidate in directory.glob(f"{species}*_communities.npy"):
        return candidate  # Return first match
    
    return None  # No file found

# === Helper Function: Stable Community Count ===
def get_stable_communities(data):
    """Calculate stable community count across runs (if multiple exist)."""
    if data.ndim == 1:  # Single run
        return len(np.unique(data))
    else:  # Multiple runs
        modes, _ = mode(data, axis=0)
        return len(np.unique(modes))

# === Compute Communities ===
records = []
missing_species = []

for _, row in eq_df.iterrows():
    species = row["Species"]
    brain_mass = row["Brain Mass (g)"]
    npy_file = find_species_file(species, npy_dir)

    if npy_file is None:
        missing_species.append(species)
        continue

    data = np.load(npy_file, allow_pickle=True)
    num_communities = get_stable_communities(data)

    records.append({
        "Species": species,
        "Brain Mass (g)": brain_mass,
        "# Communities": num_communities
    })

# === Report Missing Files ===
if missing_species:
    print(f"Warning: No files found for {len(missing_species)} species (e.g., {missing_species[:5]}...).")

# === Create DataFrame ===
df = pd.DataFrame(records)
if len(df) == 0:
    raise ValueError("No valid data found. Check file paths and naming conventions.")

df = df[df["# Communities"] > 1]  # Ensure valid communities

# Log-transform for allometric scaling
df["log_brain_mass"] = np.log10(df["Brain Mass (g)"])
df["log_communities"] = np.log10(df["# Communities"])

# === Data Quality Checks ===
print("\n=== Data Quality Checks ===")
print(f"Final sample size: {len(df)} species")
print("\nDescriptive statistics:")
print(df[["Brain Mass (g)", "# Communities"]].describe())

# Check for outliers using IQR method
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    return outliers

brain_outliers = detect_outliers(df, "Brain Mass (g)")
comm_outliers = detect_outliers(df, "# Communities")

print(f"\nPotential outliers in brain mass ({len(brain_outliers)}):")
print(brain_outliers[["Species", "Brain Mass (g)"]])

print(f"\nPotential outliers in community count ({len(comm_outliers)}):")
print(comm_outliers[["Species", "# Communities"]])

# === Correlation Analysis ===
print("\n=== Correlation Analysis ===")
# Pearson (linear correlation)
r_pearson, p_pearson = pearsonr(df["log_brain_mass"], df["log_communities"])
# Spearman (monotonic relationship)
r_spearman, p_spearman = spearmanr(df["log_brain_mass"], df["log_communities"])
# Kendall's tau (ordinal association)
tau, p_kendall = kendalltau(df["log_brain_mass"], df["log_communities"])

print(f"Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
print(f"Spearman rho = {r_spearman:.3f}, p = {p_spearman:.4f}")
print(f"Kendall's tau = {tau:.3f}, p = {p_kendall:.4f}")

# === Regression Analysis ===
print("\n=== Regression Analysis ===")
slope, intercept, r_value, p_value, _ = linregress(df["log_brain_mass"], df["log_communities"])
line_x = np.linspace(df["log_brain_mass"].min(), df["log_brain_mass"].max(), 100)
line_y = slope * line_x + intercept

print(f"Regression slope (log-log): {slope:.3f}")
print(f"R² = {r_value**2:.3f}, p = {p_value:.4f}")

# # Check for heteroscedasticity
residuals = df["log_communities"] - (slope * df["log_brain_mass"] + intercept)
# exog = sm.add_constant(df["log_brain_mass"])
# _, p_het, _, _ = het_breuschpagan(residuals, exog)

# print(f"\nHeteroscedasticity test (Breusch-Pagan): p = {p_het:.4f}")
# if p_het < 0.05:
#     print("Warning: Significant heteroscedasticity detected - regression assumptions may be violated")

# # Check for multicollinearity (though we only have one predictor)
# X = StandardScaler().fit_transform(df[["log_brain_mass"]])
# vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# print(f"\nVariance Inflation Factor (VIF): {vif[0]:.2f}")

# === Robustness Checks ===
print("\n=== Robustness Checks ===")
# 1. Check with outliers removed
df_no_outliers = df[~df.index.isin(brain_outliers.index) & ~df.index.isin(comm_outliers.index)]
if len(df_no_outliers) < len(df):
    slope_no_out, _, r_no_out, p_no_out, _ = linregress(
        df_no_outliers["log_brain_mass"], df_no_outliers["log_communities"])
    print(f"\nWithout outliers (n={len(df_no_outliers)}):")
    print(f"Slope: {slope_no_out:.3f}, R² = {r_no_out**2:.3f}, p = {p_no_out:.4f}")

# 2. Check with different community count thresholds
thresholds = [2, 3, 5]
for thresh in thresholds:
    df_thresh = df[df["# Communities"] >= thresh]
    if len(df_thresh) < len(df):
        slope_thresh, _, r_thresh, p_thresh, _ = linregress(
            df_thresh["log_brain_mass"], df_thresh["log_communities"])
        print(f"\nWith community count ≥ {thresh} (n={len(df_thresh)}):")
        print(f"Slope: {slope_thresh:.3f}, R² = {r_thresh**2:.3f}, p = {p_thresh:.4f}")

# 3. Check non-parametric relationship
print("\nNon-parametric LOWESS curve suggests:")
print("Visual inspection of the plot will show if there's a non-linear pattern")

# === Diagnostic Plots ===
plt.figure(figsize=(12, 8))

# 1. Residuals plot
plt.subplot(2, 2, 1)
plt.scatter(slope * df["log_brain_mass"] + intercept, residuals, 
            edgecolor='k', alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")

# 2. Q-Q plot
plt.subplot(2, 2, 2)
sns.regplot(x=np.sort(residuals), 
            y=np.sort(np.random.normal(0, 1, len(residuals))), 
            line_kws={'color': 'red'})
plt.xlabel("Residual quantiles")
plt.ylabel("Normal quantiles")
plt.title("Q-Q Plot")

# 3. Scale-Location plot
plt.subplot(2, 2, 3)
plt.scatter(slope * df["log_brain_mass"] + intercept, 
            np.sqrt(np.abs(residuals)), 
            edgecolor='k', alpha=0.7)
plt.xlabel("Fitted values")
plt.ylabel("√|Standardized residuals|")
plt.title("Scale-Location")

# 4. Leverage plot
plt.subplot(2, 2, 4)
influence = residuals**2 / (1 - (1/len(df) + (df["log_brain_mass"] - df["log_brain_mass"].mean())**2 / 
                              ((len(df)-1)*df["log_brain_mass"].var())))
plt.scatter(range(len(df)), influence, edgecolor='k', alpha=0.7)
plt.axhline(4/len(df), color='r', linestyle='--')  # Common cutoff
plt.xlabel("Leverage")
plt.ylabel("Influence")
plt.title("Leverage vs Influence")

plt.tight_layout()
plt.savefig(diagnostic_path, dpi=300, bbox_inches="tight")
print(f"\nSaved diagnostic plots to: {diagnostic_path}")

# === Main Plot ===
plt.figure(figsize=(10, 8))
plt.scatter(df["Brain Mass (g)"], df["# Communities"], 
            s=60, edgecolor='k', alpha=0.7, label="Species")

# Highlight outliers
for _, row in df.iterrows():
    if row["Brain Mass (g)"] > 1000 or row["# Communities"] > 20:  # Adjust thresholds as needed
        plt.text(row["Brain Mass (g)"], row["# Communities"], 
                 row["Species"], fontsize=8, ha='right')

# Regression line (log-log space)
plt.plot(10**line_x, 10**line_y, 'r--', 
         label=f'Power law: $y \sim x^{{{slope:.2f}}}$\n$R^2={r_value**2:.2f}$')

# LOWESS smooth (non-parametric)
sns.regplot(x=df["Brain Mass (g)"], y=df["# Communities"], 
            scatter=False, lowess=True, color='green', 
            line_kws={'label': 'LOWESS smooth'})

# Formatting
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Brain Mass (g, log scale)")
plt.ylabel("Number of Functional Communities (log scale)")
plt.title("Brain Mass vs. Functional Communities (MAMI Dataset)", pad=20)
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved main plot to: {output_path}")