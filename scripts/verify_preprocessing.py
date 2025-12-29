import sys
import os
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_dataset, NeuroForcastDataset

def verify_dataset(dataset_name):
    print(f"\n=== Verifying {dataset_name} ===")
    data_filename = f'train_data_{dataset_name}.npz'
    
    try:
        train_data, _, _ = load_dataset(data_filename, input_dir='./data')
        
        # Test 1: Shape before dataset
        print(f"Raw Shape: {train_data.shape}")
        
        # Initialize Dataset (should trigger preprocessing)
        ds = NeuroForcastDataset(train_data, use_graph=False, target_channels=239)
        
        # Test 2: Input Stats (after preprocess)
        # Access internal data to check values
        processed_data = ds.data # (T, C, F) list or array?
        # Actually ds.data is the full array (N, T, C, F) because preprocess returns full N array
        
        print(f"Processed Shape (Internal): {processed_data.shape}")
        
        # Check Band 0 negatives
        if processed_data.shape[3] > 0:
            band0 = processed_data[..., 0]
            print(f"Band 0 Min: {np.min(band0):.4f}, Max: {np.max(band0):.4f}")
            if np.any(band0 < 0): # Should be possible after Median scaling?
                # Wait, Robust Scaling subtracts Median. So negative values ARE expected.
                # But prior to scaling, we clamped >0.
                pass
                
        # Check overall range
        print(f"Overall Min: {np.min(processed_data):.4f}, Max: {np.max(processed_data):.4f}")
        print(f"Overall Mean: {np.mean(processed_data):.4f}, Std: {np.std(processed_data):.4f}")
        
        # Test 3: Get Item (Padding check)
        item = ds[0] # Tensor
        print(f"GetItem Shape: {item.shape}")
        
        expected_shape = (20, 239) # T, C (since use_graph=False takes feature 0)
        assert item.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {item.shape}"
        
        # Check padding
        if dataset_name == 'beignet':
            # Last channels should be 0? 
            # Original channels: 89. Padding start at 89.
            # But wait, we scaled data. Zero padding is '0'. 
            # Scaled data 0 is meaningful?
            # Zero padding happens inside __getitem__ AFTER preprocessing?
            # NO. Preprocessing happens in __init__. 
            # __getitem__ pads with 0.
            # So the padded values are exactly 0.0.
            # Let's check index 100 (should be padded)
            val_pad = item[:, 100]
            print(f"Padded Channel Index 100 Values: {val_pad.unique()}")
            assert torch.all(val_pad == 0), "Padding is not zero!"
            print("Padding verified successfully.")

    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_dataset('beignet')
    verify_dataset('affi')
