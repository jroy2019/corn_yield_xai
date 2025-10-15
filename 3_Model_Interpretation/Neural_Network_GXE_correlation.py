"""
======================================================================================================================
Script: GXE signals in Neural Network

Description:
------------
In order to detect potential GXE signals captured by the Neural Network, we compute Spearman's correlation between
SHAP contributions of SNPs and normalised weather variables across all samples.

Input:
- SHAP contribution and feature values from all fold in SNP + Weather dataset in a single pickle file.

Output:
- CSV file containing SNP SHAP x Environment feature correlations (Spearman's rho and p-value)
=======================================================================================================================
"""
# ---------------------------------------------- Import Libraries ----------------------------------------------------

import pickle
import pandas as pd
import re
from tqdm import tqdm
import sys
import numpy as np
from scipy.stats import spearmanr 

# --------------------------------------------- Correlation Computation -------------------------------------------------
# Define functions to compute correlations
def separate_features(feature_names):
    """
    Separates SNP features and environment features using regex pattern matching.
    SNP names follow pattern: S{chromosome_number}_{loci}
    """
    geno_features = [f for f in feature_names if re.match(r"S\d+_\d+", f)]
    env_features = [f for f in feature_names if f not in geno_features]
    return geno_features, env_features

def compute_shap_env_correlations(shap_values, X_test, feature_names):
    if shap_values.ndim == 3 and shap_values.shape[2] == 1:
        shap_values = np.squeeze(shap_values, axis=2)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    geno_features, env_features = separate_features(feature_names)

    corr_records = []

    for snp in tqdm(geno_features, desc="SNPs", unit="SNP"):
        for env in env_features:
            rho, pval = spearmanr(shap_df[snp], X_test[env], nan_policy="omit")
            corr_records.append((snp, env, rho, pval))

    corr_df = pd.DataFrame(corr_records, columns=["snp", "env_feature", "spearman_corr", "p_value"])
    corr_df.dropna(subset=["spearman_corr"], inplace=True)  # drop NaNs
    corr_df["abs_corr"] = corr_df["spearman_corr"].abs()
    corr_df = corr_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return corr_df

# === USER PARAMETERS ===
pkl_path = "absolute/path/to/working/directory/combined_folds.pkl"  # EDIT
output_path = "absolute/path/to/working/directory/GXE_correlation.csv"  # EDIT

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

shap_values = data["shap_values"]        
X_test = data["X_test"]                  
feature_names = data["feature_names"]    

corr_df = compute_shap_env_correlations(shap_values, X_test, feature_names)

corr_df.to_csv(output_path, index=False)

