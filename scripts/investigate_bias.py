import torch
import numpy as np
import os
import sys
from utils.data_loader import load_dataset
from model import Model

def check_bias():
    dataset_name = 'affi'
    print(f"--- Checking Calibration for {dataset_name} ---")
    
    # 1. Load Data
    data_filename = f'train_data_{dataset_name}.npz'
    _, _, val_data = load_dataset(data_filename, input_dir='./data')
    # val_data: (N, 20, C, F)
    
    # 2. Load Model
    model = Model(monkey_name=dataset_name)
    model.load()
    
    # 3. Predict
    # Mask last 10 steps
    masked_input = val_data.copy()
    masked_input[:, 10:, :, :] = masked_input[:, 9:10, :, :] 
    
    print("Predicting...")
    pred = model.predict(masked_input) 
    # pred: (N, 20, C)
    
    # 4. Compare with Ground Truth (Last 10 steps)
    gt = val_data[:, :, :, 0]
    
    res = pred - gt
    
    # Analyze Forecasting Window (10-20)
    res_future = res[:, 10:, :]
    gt_future = gt[:, 10:, :]
    pred_future = pred[:, 10:, :]
    
    bias = np.mean(res_future)
    mae = np.mean(np.abs(res_future))
    mse = np.mean(res_future**2)
    
    print(f"\nResults for {dataset_name} (Forecasting Window):")
    print(f"Bias (Mean Error): {bias:.5f}")
    print(f"MAE:  {mae:.5f}")
    print(f"MSE:  {mse:.5f}")
    
    # Ratio analysis
    # If pred is systematically smaller/larger in scale
    # ratio = mean(abs(pred)) / mean(abs(gt))
    ratio_std = np.std(pred_future) / np.std(gt_future)
    print(f"Std Ratio (Pred/GT): {ratio_std:.5f}")
    
    print("\nInterpretation:")
    if abs(bias) > 1.0: # Arbitrary threshold, depends on data scale
        print("-> Significant BIAS detected. Try Mean-Shift Calibration.")
    else:
        print("-> Bias is low.")
        
    if abs(1.0 - ratio_std) > 0.1:
        print("-> Significant SCALE difference detected. Try Scaling Calibration.")
    else:
        print("-> Scale looks okay.")

if __name__ == "__main__":
    check_bias()
