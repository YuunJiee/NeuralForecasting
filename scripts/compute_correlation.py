import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_dataset

def main():
    dataset_name = 'affi'
    data_filename = f'train_data_{dataset_name}.npz'
    input_dir = '../data'
    
    if not os.path.exists(os.path.join(input_dir, data_filename)):
        print(f"Data file {data_filename} not found.")
        return

    print("Loading data...")
    train_data, _, _ = load_dataset(data_filename, input_dir=input_dir)
    
    # train_data shape: (N, T, C, F)
    # We want C x C correlation
    N, T, C, F = train_data.shape
    
    # Flatten everything except Channels
    # (N, T, F, C) -> (-1, C)
    data_flat = train_data.transpose(0, 1, 3, 2).reshape(-1, C)
    
    print(f"Flattened shape: {data_flat.shape}")
    
    # Compute Correlation
    print("Computing correlation matrix...")
    corr_matrix = np.corrcoef(data_flat, rowvar=False)
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Handle NaNs
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Save
    os.makedirs('weights', exist_ok=True)
    save_path = f'weights/correlation_{dataset_name}.npy'
    np.save(save_path, corr_matrix)
    print(f"Saved correlation matrix to {save_path}")

if __name__ == "__main__":
    main()
