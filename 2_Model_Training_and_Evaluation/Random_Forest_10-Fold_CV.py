"""
======================================================================================================================
10-Fold Cross-Validation for Random Forest 

Description:
------------
This trains a RandomForest Regressor for corn yield prediction and implements a 10-fold cross-validation strategy. 
All samples which are the same Hybrid are grouped into the same fold and do not appear in any other fold. This facilitates 
robust evaluation of model generalization on unseen hybrids. 

Input:
This script can be run on 4 progressively enriched datasets:
- SNP only
- SNP + Weather
- SNP + Weather + Management
- SNP + Weather + Management + Stress

Model - Scikit-learn's Random Forest Regressor (with Optuna optimised hyper-parameters)  

Output:
- Overall fold metrics (RMSE, MAE, MSE, R²) saved as CSV
- Per-hybrid metrics within each fold saved as CSV

Key Features:
-------------
- 10-fold cross-validation with hybrid-based grouping
- Regression metrics (RMSE, MAE, MSE, R²) calculated per fold
- Regression metrics evaluated per hybrid group within each fold

Placeholder Explanation:
------------------------
- dataset_file: absolute path to your CSV dataset file
- dataset_name: short identifier (e.g., "snp_only", "snp_weather", etc.)
======================================================================================================================
"""

# -------------------------------------------- Import Libraries --------------------------------------------------------

import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
sys.stdout.reconfigure(line_buffering=True)

# ----------------------------------------------- CONFIGURATION ------------------------------------------------------

dataset_file = 'final_boss_snp_with_hybrids_values.csv'   # <--- CHANGE THIS
dataset_name = 'snp_only'                                 # <--- CHANGE THIS

g2f_dataset = pd.read_csv(dataset_file)

# --------------------------------------------------- Data Management ------------------------------------------------

g2f_dataset = g2f_dataset.drop(columns=['Env'])


X = g2f_dataset.drop(columns=["Yield_Mg_ha", "Hybrid"])
y = g2f_dataset["Yield_Mg_ha"]
groups = g2f_dataset["Hybrid"]

# ---------------------------------------- Cross-Validation and Model Training ---------------------------------------

gkf = GroupKFold(n_splits=10)

fold_metrics = []
per_hybrid_metrics = []

print('---------------------------------------------------------------')

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    print(f'Testing Fold {fold + 1}')

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    train_perm = np.random.permutation(len(X_train))
    X_train = X_train.iloc[train_perm]
    y_train = y_train.iloc[train_perm]

    # Train model
    rfr = RandomForestRegressor(
        n_estimators=425,    # Optuna-optimized hyperparameters
        max_features=500,    # Optuna-optimized hyperparameters
        max_samples=0.81998, # Optuna-optimized hyperparameters
        max_depth=23,        # Optuna-optimized hyperparameters
        min_samples_split=3,  # Optuna-optimized hyperparameters
        min_samples_leaf=2,   # Optuna-optimized hyperparameters
        random_state=42,
        n_jobs=-1
    )
    rfr.fit(X_train, y_train)

    y_pred = rfr.predict(X_test)

    # Baseline prediction (mean of y_train)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_rmse = math.sqrt(mean_squared_error(y_test, baseline_pred))

    # Fold metrics
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    fold_metrics.append({
        "fold": fold + 1,
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "baseline_rmse": baseline_rmse
    })

    print(f"Fold {fold + 1} RMSE: {rmse:.4f} | Baseline RMSE: {baseline_rmse:.4f}")

    # Per-hybrid metrics
    hybrids_in_fold = g2f_dataset.iloc[test_idx]["Hybrid"].values
    y_test_df = pd.DataFrame({
        "hybrid": hybrids_in_fold,
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    for hybrid_name, group in y_test_df.groupby("hybrid"):
        if len(group) < 2:
            r2_h = np.nan
        else:
            r2_h = r2_score(group["y_true"], group["y_pred"])
        rmse_h = math.sqrt(mean_squared_error(group["y_true"], group["y_pred"]))
        mae_h = mean_absolute_error(group["y_true"], group["y_pred"])
        mse_h = mean_squared_error(group["y_true"], group["y_pred"])

        per_hybrid_metrics.append({
            "fold": fold + 1,
            "hybrid": hybrid_name,
            "rmse": rmse_h,
            "mae": mae_h,
            "mse": mse_h,
            "r2": r2_h
        })

# -------------------------------------------------- Save Results -----------------------------------------------------

fold_df = pd.DataFrame(fold_metrics)
fold_df.to_csv(f"fold_metrics_{dataset_name}.csv", index=False)
print('Saved overall fold metrics')

per_hybrid_df = pd.DataFrame(per_hybrid_metrics)
per_hybrid_df.to_csv(f"per_hybrid_metrics_{dataset_name}.csv", index=False)
print('Saved per-hybrid metrics')

avg_rmse = fold_df["rmse"].mean()
print(f'Average RMSE across all folds: {avg_rmse:.4f}')

