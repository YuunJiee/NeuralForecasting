import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

def load_dataset(filename, input_dir='./data', extra_files=[], full_train=False):
    """
    Load test dataset from file.

    Args:
    Args:
        filename: Name of the test data file (e.g., 'train_data_beignet.npz')
        input_dir: Directory containing the data file
        extra_files: List of additional filenames to merge (e.g. private data)
        full_train: If True, use all data for training (no split)

    Returns:
        train_data: Samples to train
        test_data: Samples to test
        val_data: Samples to validate
    """
    test_file = os.path.join(input_dir, filename)
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Data file not found at: {test_file}")

    # Open the file and load the data
    # split into train(80%), test(10%), val(10%)
    try:
        data = np.load(test_file)['arr_0']
        
        # Load extra files
        for extra in extra_files:
            extra_path = os.path.join(input_dir, extra)
            if os.path.exists(extra_path):
                print(f"Merging extra data: {extra}")
                extra_data = np.load(extra_path)['arr_0']
                data = np.concatenate([data, extra_data], axis=0)
            else:
                print(f"Warning: Extra file {extra} not found.")
                
    except Exception as e:
        raise ValueError(f"Failed to load data. Error: {e}")
        
    if full_train:
        print(f"Full Training Mode: Using all {len(data)} samples for training.")
        train_data = data
        test_data = data # Use same data for test/val to prevent errors, but ignore metrics
        val_data = data
    else:
        train_end = int(len(data) * 0.8)
        test_end = int(len(data) * 0.9)
        
        train_data = data[:train_end]
        test_data = data[train_end:test_end]
        val_data = data[test_end:]

    return train_data, test_data, val_data

def apply_gaussian_blur(data, sigma=1.0):
    """
    Apply Gaussian Blur to smooth the data along the time axis.
    Args:
        data: Numpy array of shape (N, T, C, F)
        sigma: Standard deviation for Gaussian kernel
    Returns:
        blurred_data: Numpy array of same shape
    """
    if sigma <= 0:
        return data
    
    print(f"Applying Gaussian Blur with sigma={sigma}...")
    # Apply along axis 1 (Time)
    return gaussian_filter1d(data, sigma=sigma, axis=1)


def normalize(data, average=[], std=[]):
    """
    Normalizes input to the range of [~mean - 4*std, ~mean + 4*std] mapped to [-1, 1].
    
    Args:
        data: Input data array
        average: Precomputed average (optional)
        std: Precomputed std (optional)
        
    Returns:
        norm_data: Normalized data
        average: Computed or used average
        std: Computed or used std
    """
    # because in this dataset, the timestep is in first dimension (relative to sample?). 
    # Actually based on shape (N, T, C, F), logic below handles N*T flattening.
    # adapt the normalization to fit this situation, compute mean and std in axis 0
    
    n = data.shape[0]
    t = data.shape[1]
    c = data.shape[2]
    f = data.shape[3] if data.ndim == 4 else 1

    original_shape = data.shape
    
    # --- OPTIMIZATION 1: Band 0 Negative Clamping ---
    # REVERTED: Caused R2 drop to 0.52. Negative values are significant.
    # if data.ndim == 4 and data.shape[3] >= 1:
    #     band0 = data[..., 0]
    #     band0[band0 < 0] = 0
    #     data[..., 0] = band0
        
    if data.ndim == 4:
        # n, t, c, f = data.shape already defined
        data_reshaped = data.reshape((n*t, -1))  # neuron input flattened
    else:
        # Handle cases where data might already be reshaped or different dims?
        # Assuming strict adherence to demo logic for now
        data_reshaped = data.reshape((n*t, -1))

    if len(average) == 0:
        average = np.mean(data_reshaped, axis=0, keepdims=True)
        std = np.std(data_reshaped, axis=0, keepdims=True)
        
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    
    # Avoid division by zero
    denominator = combine_max - combine_min
    denominator[denominator == 0] = 1e-8
    
    norm_data = 2 * (data_reshaped - combine_min) / denominator - 1
    norm_data = norm_data.reshape(original_shape)
    
    return norm_data, average, std


class NeuroForcastDataset(Dataset):
    def __init__(self, neural_data, use_graph=False, average=[], std=[]):
        """
        Args:
            neural_data: N*T*C*F
            use_graph: Boolean
            average: Precomputed mean
            std: Precomputed std
        """
        self.data = neural_data
        self.use_graph = use_graph
        
        # Normalize data upon initialization
        if len(average) == 0:
            self.data, self.average, self.std = normalize(self.data)
        else:
            self.data, self.average, self.std = normalize(self.data, average, std)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
        if not self.use_graph:
            # Take only the first feature (index 0 on last dimension?)
            # Demo says: data = data[:, :, 0] inside getitem? 
            # Wait, self.data[index] returns (T, C, F)
            # So data[:, :, 0] means (T, C)
            data = data[:, :, 0]

        # Convert to tensor
        data = torch.tensor(data, dtype=torch.float32)
        return data
