"""
======================================================================================================================
Random Forest Global Feature Importances

Description:
-------------
A Random Forest trained from scratch and global feature importances computed per fold 
using feature_importances_ attribute in scikit-learn RandomForestRegressor.

Input:
This script can be run on 4 progressively enriched datasets:
- SNP only
- SNP + Weather
- SNP + Weather + Management
- SNP + Weather + Management + Stress

Model - Scikit-learn's Random Forest Regressor (with Optuna optimised hyper-parameters)

Output:
- Pickle file with trained Random Forest models.
- CSV file containing built-in Random Forest global feature importances per fold


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
import pickle
from pathlib import Path
import sys

sys.stdout.reconfigure(line_buffering=True)

# -------------------------------------------- Configuration  -------------------------------------------------------

dataset_file = 'final_boss_snp_with_hybrids_values.csv'   # <--- CHANGE THIS
dataset_name = 'snp_only'                                 # <--- CHANGE THIS

g2f_dataset = pd.read_csv(dataset_file)

# ------------------------------------------------ Data Import + Management ----------------------------------------

g2f_dataset = pd.read_csv(dataset_file)

groups = g2f_dataset["Hybrid"]
X = g2f_dataset.drop(columns=["Yield_Mg_ha", "Hybrid", "Env"])
y = g2f_dataset["Yield_Mg_ha"]

# ----------------------------------------- Model Training and RF Global Importances --------------------------------
save_dir = Path(f"absolute/path/to/working/directory/rf_global_feature_importance_{dataset_name}")
save_dir.mkdir(parents=True, exist_ok=True)

gkf = GroupKFold(n_splits=10)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f'Processing Fold {fold + 1}')

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_perm = np.random.permutation(len(X_train))
    X_train = X_train.iloc[train_perm]
    y_train = y_train.iloc[train_perm]

    # Train Random Forest
    rfr = RandomForestRegressor(
        n_estimators=425,      # Optuna-optimised
        max_features=500,      # Optuna-optimised
        max_samples=0.81998,   # Optuna-optimised
        max_depth=23,          # Optuna-optimised
        min_samples_split=3,   # Optuna-optimised
        min_samples_leaf=2,    # Optuna-optimised
        random_state=42,
        n_jobs=-1
    )
    rfr.fit(X_train, y_train)

    # Save model 
    model_save_path = save_dir / f"fold_{fold+1}_model.pkl"
    with open(model_save_path, "wb") as f:
        pickle.dump(rfr, f)
    print(f"Fold {fold+1} model saved.")

    # Save RF feature importances 
    rf_importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "rf_importance": rfr.feature_importances_
    }).sort_values("rf_importance", ascending=False)
    csv_save_path = save_dir / f"fold_{fold+1}_rf_importance.csv"
    rf_importance_df.to_csv(csv_save_path, index=False)


