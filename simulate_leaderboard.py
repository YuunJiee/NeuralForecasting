import os
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from model import Model
from utils.data_loader import load_dataset, NeuroForcastDataset

def evaluate_dataset(model, data, dataset_name, is_private=False):
    """
    Evaluate model on a specific dataset.
    """
    print(f"Evaluating on {dataset_name} (Private={is_private})...")
    
    # Preprocessing
    # If using NeuroForcastDataset logic, we need to manually handle it because
    # Model.predict expects raw numpy array (N, T, C, F)
    
    # Check data shape
    # data shape: (N, T, C, F)
    
    # If private file, we treat the whole file as test set
    # If public (train_data), we passed the validation split
    
    # Model.predict handles normalization internally using loaded stats.
    # So we just pass raw data.
    
    # Run prediction
    try:
        # Predict
        # Input to predict: (N, 20, C, F)
        # It returns (N, 20, C)
        predictions = model.predict(data)
        
        # Ground Truth
        # data is (N, 20, C, F) -> Take first feature and steps 10-19
        gt = data[:, 10:, :, 0] # (N, 10, C)
        
        # Preds are (N, 20, C) -> Take steps 10-19
        preds = predictions[:, 10:, :] # (N, 10, C)
        
        # Calculate MSE
        mse = np.mean((preds - gt) ** 2)
        return mse
        
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {e}")
        return float('nan')

def main():
    print("--- Simulating Leaderboard Evaluation ---")
    
    # 1. Initialize Models
    print("Loading Models (with blur_sigma=1.0)...")
    model_beignet = Model('beignet', blur_sigma=1.0)
    model_beignet.load()
    
    model_affi = Model('affi', blur_sigma=1.0)
    model_affi.load()
    
    # 2. Load Datasets
    data_dir = './data'
    
    # --- Public Datasets (Validation Split) ---
    print("\nLoading Public Datasets...")
    _, _, val_affi = load_dataset('train_data_affi.npz', input_dir=data_dir)
    _, _, val_beignet = load_dataset('train_data_beignet.npz', input_dir=data_dir)
    
    # --- Private Datasets (Full File) ---
    print("Loading Private Datasets...")
    
    def load_private(filename):
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return np.load(path)['arr_0']
        else:
            print(f"Warning: {filename} not found.")
            return None

    affi_d2 = load_private('train_data_affi_2024-03-20_private.npz')
    beignet_d2 = load_private('train_data_beignet_2022-06-01_private.npz')
    beignet_d3 = load_private('train_data_beignet_2022-06-02_private.npz')
    
    # 3. Evaluate
    results = {}
    
    # MSE affi (Public)
    results['MSE affi'] = evaluate_dataset(model_affi, val_affi, 'Affi Public')
    
    # MSE beignet (Public)
    results['MSE beignet'] = evaluate_dataset(model_beignet, val_beignet, 'Beignet Public')
    
    # MSE affi D2
    if affi_d2 is not None:
        results['MSE affi D2'] = evaluate_dataset(model_affi, affi_d2, 'Affi D2 (Private)', is_private=True)
    else:
        results['MSE affi D2'] = float('nan')
        
    # MSE beignet D2
    if beignet_d2 is not None:
        results['MSE beignet D2'] = evaluate_dataset(model_beignet, beignet_d2, 'Beignet D2 (Private)', is_private=True)
    else:
        results['MSE beignet D2'] = float('nan')
        
    # MSE beignet D3
    if beignet_d3 is not None:
        results['MSE beignet D3'] = evaluate_dataset(model_beignet, beignet_d3, 'Beignet D3 (Private)', is_private=True)
    else:
        results['MSE beignet D3'] = float('nan')
        
    # 4. Calculate Total MSR
    # Average of available scores
    valid_scores = [v for v in results.values() if not np.isnan(v)]
    if valid_scores:
        total_msr = sum(valid_scores) / len(valid_scores)
    else:
        total_msr = float('nan')
        
    results['Total MSR'] = total_msr
    
    # 5. Display Table
    print("\n" + "="*80)
    print("LEADERBOARD SIMULATION")
    print("="*80)
    
    # Create DataFrame for nice printing
    df = pd.DataFrame([results])
    
    # Reorder columns to match user request
    cols = ['MSE affi', 'MSE beignet', 'MSE affi D2', 'MSE beignet D2', 'MSE beignet D3', 'Total MSR']
    df = df[cols]
    
    # Add ID and Date
    df.insert(0, 'ID', 'Current_Model')
    df.insert(0, 'Date', datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()
