"""
======================================================================================================================
Script: SHAP Local Explanations of Random Forest holdout predictions

Description:
------------
Use SHAP TreeExplainer to interpret indivual yield prediction of 10,000 samples
from a Random Forest model.

Input:
This script can be run on 4 progressively enriched datasets:
- SNP only
- SNP + Weather
- SNP + Weather + Management
- SNP + Weather + Management + Stress

Model:
- Scikit-learn Random Forest Regressor with Optuna-optimised hyperparameters

Output (per fold):
------------------
- Pickle file of the  trained Random Forest model 
- Pickle file containing:
    * Feature names
    * Feature values 
    * Original sample index (10,000 samples)
    * SHAP values
    * Expected value

Placeholder Explanation:
------------------------
- dataset_file: absolute path to your CSV dataset file
- dataset_name: short identifier (e.g., "snp_only", "snp_weather", etc.)
======================================================================================================================
"""
# ---------------------------------------------- Import Libraries ----------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import shap
import pickle
from pathlib import Path
import sys
sys.stdout.reconfigure(line_buffering=True)


# ------------------------------------------------ Configuration ---------------------------------------------------
filepath = '/absolute/path/to/working/directory/'  
dataset_file = f'{filepath}/final_boss_snp_with_hybrids_values.csv' # <--- CHANGE THIS!
dataset_name = 'snp_only'                                                                        # <--- CHANGE THIS!

# ------------------------------------------------ Data Import + Management ------------------------------------------
g2f_dataset = pd.read_csv(dataset_file)

groups = g2f_dataset["Hybrid"]
X = g2f_dataset.drop(columns=["Yield_Mg_ha", "Hybrid", "Env"])
y = g2f_dataset["Yield_Mg_ha"]

# --------------------- Model Training and Local SHAP Importance (1000 samples per fold) ---------------------------
save_dir = Path(f"{filepath}/shap_results_local_predictions_{dataset_name}")
save_dir.mkdir(parents=True, exist_ok=True)

gkf = GroupKFold(n_splits=10)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_perm = np.random.permutation(len(X_train))
    X_train = X_train.iloc[train_perm]
    y_train = y_train.iloc[train_perm]

    rfr = RandomForestRegressor(
        n_estimators=425, # Optuna-optimized 
        max_features=500, # Optuna-optimized 
        max_samples=0.81998, # Optuna-optimized 
        max_depth=23, # Optuna-optimized 
        min_samples_split=3, # Optuna-optimized 
        min_samples_leaf=2, # Optuna-optimized 
        random_state=42,
        n_jobs=-1
    )
    rfr.fit(X_train, y_train)

    # Save trained model
    model_save_path = save_dir / f"fold_{fold+1}_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(rfr, f)
    print(f"Fold {fold+1} model saved.")

    # -------------------- SHAP Local Explanations -------------------------
    explainer = shap.TreeExplainer(rfr, feature_perturbation="auto")

    # Sample test set to speed up SHAP calculations
    sample_size = min(len(X_test), 1000)  
    X_test_sampled = X_test.sample(sample_size, random_state=42)

    shap_values = explainer.shap_values(X_test_sampled)

    # Save SHAP values
    output_file = save_dir / f"fold_{fold+1}_1000_samples_shap_values.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "index": X_test_sampled.index.tolist(),
            "X_test": X_test_sampled,
            "shap_values": shap_values,
            "feature_names": X_test_sampled.columns.tolist(),
            "expected_value": explainer.expected_value
        }, f)

