"""
======================================================================================================================
10-Fold Cross-Validation for Neural Network 

Description:
------------
This trains a Fully Connected Neural Network for corn yield prediction and implements a 10-fold cross-validation strategy. 
All samples which are the same Hybrid are grouped into the same fold and do not appear in any other fold. This facilitates 
robust evaluation of model generalization on unseen hybrids. 

Input: 
This script can be run on 4 progressively enriched datasets:
- SNP only
- SNP + Weather
- SNP + Weather + Management
- SNP + Weather + Management + Stress

Output: 
- CSV files for each fold, containing train loss, test loss, and learning rate per epoch
- Best model parameters saved for each fold in .pth file
- CSV file containing regression metrics (RMSE, MAE, MSE, R²) per fold
- CSV file containing regression metrics (RMSE, MAE, MSE, R²) per hybrid within each fold

Model - Fully Connected Neural Network using PyTorch (with Optuna optimised hyper-parameters)

Key Features:
-------------
- Convert Data to PyTorch Dataset and DataLoader objects
- 10-fold cross-validation with hybrid-based grouping
- Neural Network training pipeline includes:
    * Exponential learning rate decay 
    * Early stopping based on test loss  

Placeholders:
-------------
- dataset_name: short name/description of the dataset version (e.g., "snp_weather")
- dataset_file: absolute path to your CSV dataset file
======================================================================================================================
"""

# --------------------------------------------------- Import Libraries and Data --------------------------------------------------------
 
import pandas as pd
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from timeit import default_timer as timer
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------- CONFIGURATION ---------------------------------------------------

dataset_file = 'final_boss_snp_with_hybrids_values.csv'   # <--- CHANGE THIS
dataset_name = 'snp_only'                                 # <--- CHANGE THIS

g2f_dataset = pd.read_csv(dataset_file)

# ------------------------------------------------ DATA MANAGEMENT ------------------------------------------------ 

# Ensure Weather features are normalised (troubleshooting previous mistake)
if dataset_name != 'snp_only':
    cols_to_normalize = g2f_dataset.columns[2159:2537]
    scaler = MinMaxScaler()
    g2f_dataset.loc[:, cols_to_normalize] = scaler.fit_transform(g2f_dataset[cols_to_normalize])

hybrids = g2f_dataset['Hybrid'].values

g2f_dataset = g2f_dataset.drop(columns=['Env', 'Hybrid'])
features = g2f_dataset.drop(columns=['Yield_Mg_ha']).to_numpy()
target = g2f_dataset['Yield_Mg_ha'].to_numpy()

# ------------------------------------------------------- Model Construction ----------------------------------------------------------

class RegressionModelV0(nn.Module):
    def __init__(self, input_features, output_features=1, hidden_units_1=128, hidden_units_2=64, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_units_1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_1),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units_1, hidden_units_2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units_2),
            nn.Linear(hidden_units_2, output_features)
        )

    def forward(self, x):
        return self.network(x)

print('Model has been successfully constructed')

# -------------------------------------- Construct a custom Pytorch Dataset Class  ----------------------------------

class G2FDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# ----------------------------------------- Define custom training functions -------------------------------------

def train_step(model, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0
    for X, y in dataloader:
        y_preds = model(X).squeeze(1)
        loss = loss_fn(y_preds, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(dataloader)

def test_step(model, dataloader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X, y in dataloader:
            test_preds = model(X).squeeze(1)
            loss = loss_fn(test_preds, y)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs=100,
          patience=10, min_delta=0.001, checkpoint_path='best_model.pth'):
    results = {"train_loss": [], "test_loss": [], "lr": []}
    best_loss = float("inf")
    patience_counter = 0
    lr_lambda = lambda epoch: 1.0 if epoch <= 19 else 0.9 ** (epoch - 19)  
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer)
        test_loss = test_step(model, test_dataloader, loss_fn)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, lr={current_lr:.6f}")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["lr"].append(current_lr)

        scheduler.step()

        if best_loss - test_loss > min_delta:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved improved model at epoch {epoch+1} with test loss {test_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        print('------------------------------------')

    return results

# ------------------------------------------- Cross Validation and Model Training ----------------------------

NUM_EPOCHS = 100
BATCH_SIZE = 32

loss_fn = nn.MSELoss()   # CHANGE TO OPTIMISED LOSS FUNCTION
group_kfold = GroupKFold(n_splits=10)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(group_kfold.split(features, target, groups=hybrids)):
    print(f"\n========================= FOLD {fold+1}/10 =========================")
 
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train = features[train_idx], target[train_idx]
    X_test, y_test = features[test_idx], target[test_idx]

    train_dataset = G2FDataset(X_train, y_train)
    test_dataset = G2FDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RegressionModelV0(input_features=features.shape[1], 
                              hidden_units_1=72,    # CHANGE TO OPTIMISED PARAM 
                              hidden_units_2=96,   # CHANGE TO OPTIMISED PARAM 
                              output_features=1,
                              dropout_rate=0.3)     # CHANGE TO OPTIMISED PARAM 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # CHANGE TO OPTIMISED OPTIMIZER

    checkpoint_path = f'model_fold_{fold+1}_{dataset_name}.pth'
    
    start_time = timer()
        
    print(f'{fold + 1}/10 started training')

    result = train(model, train_loader, test_loader, optimizer, loss_fn,
                   epochs=NUM_EPOCHS,
                   patience=10,
                   min_delta=0.001,
                   checkpoint_path=checkpoint_path)
    end_time = timer()
        
    print(f"{fold +1}/10 training completed. Total training time: {end_time - start_time:.2f} seconds")
    
    fold_results.append(result)

    pd.DataFrame(result).to_csv(f"fold_{fold+1}_results_{dataset_name}.csv", index=False)

    # --- Restore best model and save the final state dict ---
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    final_model_path = f"model_fold_{fold+1}_state_dict_{dataset_name}.pth"
    torch.save(model.state_dict(), final_model_path)

# -------------------------------------------- Model Evaluation -----------------------------------

results = []
per_hybrid_metrics = []

for fold, (_, test_idx) in enumerate(group_kfold.split(features, target, groups=hybrids)):
    print(f"\n========================= FOLD {fold+1}/10 =========================")

    X_test, y_test = features[test_idx], target[test_idx]
    test_dataset = G2FDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RegressionModelV0(input_features=features.shape[1], 
                              hidden_units_1=104,    # CHANGE TO OPTIMISED PARAM 
                              hidden_units_2=88,   # CHANGE TO OPTIMISED PARAM 
                              output_features=1,
                              dropout_rate=0.2)     # CHANGE TO OPTIMISED PARAM 
    
    model_path = f"model_fold_{fold+1}_state_dict_{dataset_name}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds_list = []
    targets_list = []

    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            preds_list.extend(outputs.numpy())
            targets_list.extend(y_batch.numpy())

    preds = np.array(preds_list)
    targets = np.array(targets_list)

    # Calculate Regression Metrics for the fold
    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    print(f"RMSE: {rmse:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    results.append({
        "Fold": fold+1,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    })
    
    # Calculate Regression Metrics per Hybrid
    hybrids_series = g2f_dataset['Hybrid']
    hybrids_in_fold = hybrids_series.iloc[test_idx].values

    y_test_df = pd.DataFrame({
        "hybrid": hybrids_in_fold,
        "y_true": targets,
        "y_pred": preds
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

print("All folds tested")

# -------------------------------------------------- Save Results -----------------------------------------------------

results_df = pd.DataFrame(results)
pd.DataFrame(per_hybrid_metrics).to_csv(f"nn_per_{dataset_name}_hybrid_metrics.csv", index=False)
    


