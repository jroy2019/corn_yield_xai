"""
======================================================================================================================
Script: SHAP Local and Global Explanations of Neural Network of holdout predictions

Description:
------------
Use SHAP DeepExplainer to interpret indivual yield predictions from a pre-trained
Neural Network and compute mean absolute SHAP value per feature across all 
samples for global feature importance.

Input:
This script can be run on 4 progressively enriched datasets:
- SNP only
- SNP + Weather
- SNP + Weather + Management
- SNP + Weather + Management + Stress

Model:
- Pytorch Fully Connect Neural network with non-linear activations (with Optuna optimised hyper-parameters)

Output:
- Pickle (per fold) file containing: 
    * Feature names
    * Feature values 
    * Original Sample index
    * SHAP values
    * Expected value

Placeholder Explanation:
------------------------
- dataset_file: absolute path to your CSV dataset file
- dataset_name: short identifier (e.g., "snp_only", "snp_weather", etc.)
=======================================================================================================================
"""
# ---------------------------------------------- Import Libraries ----------------------------------------------------

import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
import shap
import pickle
import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

# -------------------------------------------- Configuration  -------------------------------------------------------

filepath = 'absolute/path/to/working/directory/'  # <--- CHANGE THIS!
dataset_file = f'{filepath}final_boss_snp_with_hybrids_values.csv'   # <--- CHANGE THIS!
dataset_name = 'snp_only'                                           # <--- CHANGE THIS!


# ------------------------------------------------ Data Import + Management ------------------------------------------

g2f_dataset = pd.read_csv(dataset_file)

# Troubleshooting: I forgot to normalise weather features in final dataset
if dataset_name != 'snp_only': 
    cols_to_normalize = g2f_dataset.columns[2159:2537]
    scaler = MinMaxScaler()
    g2f_dataset[cols_to_normalize] = scaler.fit_transform(g2f_dataset[cols_to_normalize])

groups = g2f_dataset['Hybrid'].values
g2f_dataset = g2f_dataset.drop(columns=['Env', 'Hybrid'])
X = g2f_dataset.drop(columns=['Yield_Mg_ha'])
y = g2f_dataset['Yield_Mg_ha']

# ----------------------------------------- Model Construction -------------------------------------------------------

class RegressionModelV0(nn.Module):
    def __init__(self, input_features, output_features=1, hidden_units_1=128, hidden_units_2=64, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units_1),
            nn.ReLU(), #non-linear activation
            nn.BatchNorm1d(hidden_units_1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(), #non-linear activation
            nn.BatchNorm1d(hidden_units_2),
            nn.Linear(hidden_units_2, output_features)
        )

    def forward(self, x):
        return self.network(x)

# ------------------------------------------- Local SHAP explanations -------------------------------------------
save_dir = Path(f"{filepath}/shap_local_predictions_{dataset_name}")
save_dir.mkdir(parents=True, exist_ok=True)

gkf = GroupKFold(n_splits=10)
RANDOM_STATE = 42 
model = RegressionModelV0(input_features=X.shape[1], 
                          hidden_units_1=64,    # CHANGE TO OPTIMISED PARAM 
                          hidden_units_2=112,   # CHANGE TO OPTIMISED PARAM 
                          output_features=1,
                          dropout_rate=0.3)     # CHANGE TO OPTIMISED PARAM 

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    # Load Neural Network 
    model_path = f"{filepath}/model_fold_{fold+1}_state_dict_{dataset_name}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    original_indices = X_test.index.tolist()
    feature_names = X_test.columns.tolist()
 
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
    
    # Background: small random subset of X_train 
    background = torch.tensor(
        X_train.sample(n=100, random_state=RANDOM_STATE).values,
        dtype=torch.float
    )

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test_tensor)
    expected_value = explainer.expected_value

    # Save individual SHAP values + metadata in pickle format
    output_file = save_dir / f"fold_{fold+1}_all_samples_shap_values.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({
            "index": original_indices,
            "X_test": X_test,
            "shap_values": shap_values,
            "feature_names": feature_names,
            "expected_value": expected_value
        }, f)
# ------------------------------------------- Global SHAP explanations -------------------------------------------
num_folds = 10

for fold in range(num_folds):
    pickle_file = filepath / f"shap_local_predictions_{dataset_name}/fold_{fold+1}_all_samples_shap_values.pkl"
     
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    shap_values = data["shap_values"]                              
    feature_names = data["feature_names"]
    shap_values = shap_values.squeeze(-1)

    # Compute mean absolute SHAP values per feature
    mean_abs_importance = np.abs(shap_values).mean(axis=0)

    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_value": mean_abs_importance
    }).sort_values(by="mean_abs_shap_value", ascending=False).reset_index(drop=True)

    output_file = filepath / f"shap_global_predictions_{dataset_name}/fold_{fold+1}_shap_global_importance_{dataset_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_importance.to_csv(output_file, index=False)

