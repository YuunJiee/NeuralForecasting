import numpy as np
import torch

def get_pearson_correlation(data):
    """
    Compute Pearson Correlation Matrix for the given data.
    
    Args:
        data: Numpy array of shape (N, T, C, F) or similar. 
              We want correlation between Channels (C).
              We will reshape to (N*T, C) to compute correlation between C variables over all time steps.
    
    Returns:
        adj: Pearson Correlation Matrix of shape (C, C).
    """
    # Check dimensions
    if data.ndim == 4: # (N, T, C, F)
        n, t, c, f = data.shape
        # Flatten N and T, and assume we correlate based on first feature if multiple
        # Or better: correlate based on the flattened 0-th feature which is the main signal
        data_flat = data[:, :, :, 0].reshape(n * t, c)
    elif data.ndim == 3: # (N, T, C)
        n, t, c = data.shape
        data_flat = data.reshape(n * t, c)
    else:
        raise ValueError(f"Unsupported data shape for correlation: {data.shape}")

    # Compute correlation
    # rowvar=False means each column is a variable (Channel)
    corr_matrix = np.corrcoef(data_flat, rowvar=False) 
    
    # Handle NaNs if any constant columns exist
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return torch.tensor(corr_matrix, dtype=torch.float32)
