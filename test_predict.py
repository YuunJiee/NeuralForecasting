import torch
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import load_dataset
from model import Model

def r2_score_np(target_data, output):
    # Avoid division by zero
    denominator = np.sum(target_data ** 2)
    if denominator == 0:
        return 0.0
    return 1 - (np.sum((target_data - output) ** 2) / denominator)

def mse_score_np(target_data, output):
    return np.mean((target_data - output) ** 2)

def prepare_test_input(data, init_steps=10):
    """
    Simulate the competition input by masking the last 10 steps.
    data: (N, 20, C, F)
    Returns: (N, 20, C, F) with last 10 steps masked (repeated from step 10)
    """
    masked_data = data.copy()
    
    # Iterate over samples to be safe (or use broadcasting)
    # data shape: (N, T, C, F)
    # Step 9 is the 10th step (0-indexed)
    
    # Repeat the 10th step (index 9) for indices 10 to 19
    # masked_data[:, 10:, :, :] = masked_data[:, 9:10, :, :] 
    # But wait, we need to repeat properly.
    
    last_known_step = masked_data[:, 9:10, :, :] # Shape (N, 1, C, F)
    
    # Broadcast/Tile to fill (N, 10, C, F)
    # We can just assign using broadcasting if numpy supports it
    masked_data[:, 10:, :, :] = last_known_step
    
    return masked_data

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beignet', choices=['beignet', 'affi'], help='Dataset name')
    parser.add_argument('--model', type=str, default=None, choices=['amag', 'dlinear_gnn', 'stndt', 'dlinear_stndt'], help='Model architecture')
    args = parser.parse_args()

    print("--- Running Model Self-Test ---")
    
    subject = args.dataset
    model_type = args.model
    # Correcting the print statement syntax and variable usage
    print(f"Initializing Model for {subject} ({model_type})...")

    # --- Configuration ---
    dataset_name = args.dataset 
    print(f"Evaluating for dataset: {dataset_name}")

    # --- Data Loading ---
    data_filename = f'train_data_{dataset_name}.npz'
    if not os.path.exists(os.path.join('./data', data_filename)):
        print(f"Error: Data file {data_filename} not found.")
        return

    print("Loading validation data...")
    # We only care about validation data for testing performance
    _, _, val_data = load_dataset(data_filename, input_dir='./data')
    print(f"Validation set shape: {val_data.shape}") # (N, 20, C, F)

    # --- Model Loading ---
    print("Loading model...")
    try:
        model = Model(monkey_name=dataset_name, model_type=model_type)
        model.load()
    except Exception as e:
        print(f"Failed to initialize/load model: {e}")
        return

    # --- Preparation ---
    # Important: The model should not see the ground truth of the last 10 steps.
    # In the competition, inputs are masked. We must simulate this.
    print("Preparing masked input (hiding future steps)...")
    masked_input = prepare_test_input(val_data)
    
    # --- Prediction ---
    print("Running prediction...")
    # Model returns (N, 20, C)
    predictions_full = model.predict(masked_input)
    
    # Ground Truth: Take the first feature (as per demo logic) 
    # val_data is (N, 20, C, F). Target is val_data[:, :, :, 0]
    ground_truth_full = val_data[:, :, :, 0]

    # --- Evaluation ---
    # We evaluate mainly on the FORECASTING part (last 10 steps)
    # But we can also show full sequence metrics.
    
    init_steps = 10
    
    # 1. Forecasting Metrics (Last 10 steps)
    pred_future = predictions_full[:, init_steps:, :]
    gt_future = ground_truth_full[:, init_steps:, :]
    
    mse_future = mse_score_np(gt_future, pred_future)
    r2_future = r2_score_np(gt_future, pred_future)
    
    # 2. Full Sequence Metrics
    mse_full = mse_score_np(ground_truth_full, predictions_full)
    r2_full = r2_score_np(ground_truth_full, predictions_full)

    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Forecasting (Last 10 steps):")
    print(f"  MSE: {mse_future:.6f}")
    print(f"  R2 : {r2_future:.6f}")
    print("-" * 30)
    print(f"Full Sequence (20 steps):")
    print(f"  MSE: {mse_full:.6f}")
    print(f"  R2 : {r2_full:.6f}")
    print("="*30)

    # Sanity Check
    if r2_future < 0:
        print("\nNote: Negative R2 implies the model is performing worse than a horizontal line (mean).")

if __name__ == "__main__":
    main()
