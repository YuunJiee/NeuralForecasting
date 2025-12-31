import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_dataset
from model import Model

def mse_score_np(target_data, output):
    return np.mean((target_data - output) ** 2)

def experiment_smoothing():
    print("--- Experimenting with Inference Post-Processing ---")
    
    datasets = ['affi', 'beignet']
    
    for dataset_name in datasets:
        print(f"\ndataset: {dataset_name}")
        
        # 1. Load Data
        data_filename = f'train_data_{dataset_name}.npz'
        _, _, val_data = load_dataset(data_filename, input_dir='./data')
        # val_data: (N, 20, C, F)
        
        # 2. Load Model
        model = Model(monkey_name=dataset_name)
        model.load()
        
        # 3. Predict (Base)
        masked_input = val_data.copy()
        masked_input[:, 10:, :, :] = masked_input[:, 9:10, :, :] 
        
        print("Predicting Base...")
        pred_base = model.predict(masked_input) # Already has bias fix for Affi if in model.py
        
        gt = val_data[:, :, :, 0]
        
        # Evaluate Base
        mse_base = mse_score_np(gt[:, 10:, :], pred_base[:, 10:, :])
        print(f"Base MSE (Last 10): {mse_base:.5f}")
        
        # --- Experiment 1: Temporal Smoothing (EMA) ---
        # pred[t] = alpha * pred[t] + (1-alpha) * pred[t-1]
        # We smooth along axis 1 (Time).
        # Note: pred_base includes first 10 steps (history). We smoothing whole seq?
        # Usually smoothing helpful for high frequency noise.
        
        best_mse_smooth = mse_base
        best_alpha = 1.0
        
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_smooth = pred_base.copy()
            # Apply simple EMA on the PREDICTION part (10-19)
            # We initialize t=10 with pred[10]
            for t in range(11, 20):
                pred_smooth[:, t, :] = alpha * pred_base[:, t, :] + (1 - alpha) * pred_smooth[:, t-1, :]
            
            mse_smooth = mse_score_np(gt[:, 10:, :], pred_smooth[:, 10:, :])
            if mse_smooth < best_mse_smooth:
                best_mse_smooth = mse_smooth
                best_alpha = alpha
                
        print(f"Smoothing Best MSE: {best_mse_smooth:.5f} (alpha={best_alpha})")
        print(f"  -> Improvement: {mse_base - best_mse_smooth:.5f}")

        # --- Experiment 2: Session-Aware Bias Correction ---
        # Adjust prediction based on the mean difference in the INPUT window (0-9)
        # bias_local = mean(pred_hist) - mean(input_hist)
        # pred_new = pred - bias_local
        # Idea: If model predicts history incorrectly (shifted), it likely shifts future too.
        # But wait, model.predict returns history as input copy mostly.
        # Let's check model internal reconstruction if available? 
        # Actually standard model.predict returns input copy for first 10 steps.
        # So pred_base[:, :10, :] == masked_input[:, :10, :, 0] roughly.
        
        # Alternative Session-Aware:
        # Match the "Trend" at step 9.
        # gap = pred[10] - input[9]
        # if gap is too huge, maybe dampen it?
        
        # Let's try: Aligning mean of prediction with mean of last k input steps?
        # Maybe scale adjustment?
        best_mse_scale = mse_base
        best_factor = 1.0
        
        # Simple global scaling per session?
        # ratio = std(input) / std(pred_future) ?
        pass

if __name__ == "__main__":
    experiment_smoothing()
