import torch
import torch.nn as nn
import os
import numpy as np

# Define the baseline model from the demo notebook
class NFBaseModel(nn.Module):
    def __init__(self, input_size=96, hidden_size=256):
        super(NFBaseModel, self).__init__()
        self.encoder = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (Batch, Time, Features)? 
        # Based on demo, input x seems to be (Batch, Time, Channel) effectively if use_graph=False
        # But wait, dataset returns shape (B, T, C). 
        # Let's check demo again:
        # data = data[:, :, 0] if not use_graph (taking first feature only?)
        # data shape in notebook: (B, T, C, F) -> normalized -> (B, T, C, F)
        # NFBaseModel input_size=num_channels.
        # So it seems the model processes time series of 'channels'.
        
        output, hidden = self.encoder(x)
        output = self.output_layer(output)
        return output

class Model:
    def __init__(self, monkey_name=""):
        """
        Initialize the model wrapper.
        Args:
            monkey_name (str): Name of the subject ('beignet' or 'affi').
        """
        self.monkey_name = monkey_name
        
        # Configuration based on monkey_name
        if self.monkey_name == 'beignet':
            self.input_size = 89
        elif self.monkey_name == 'affi':
            self.input_size = 239
        else:
            # Default fallback for testing or empty initialization
            if self.monkey_name == "":
                # Attempt to guess based on available weights? Or just default.
                # For submission safety, let's default to beignet if empty, 
                # but print a warning.
                # print("Warning: monkey_name not specified. Defaulting to 'beignet'.")
                self.input_size = 89
            else:
                raise ValueError(f'No such a monkey: {self.monkey_name}')
        
        # Hyperparameters from demo
        self.hidden_size = 1024 
        # Note: Demo used hidden_size=1024 in the main block, but class def default was 256. 
        # We use 1024 to match the demo execution script.

        # Initialize the actual PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NFBaseModel(input_size=self.input_size, hidden_size=self.hidden_size)
        self.model.to(self.device)
        
        # Statistics for normalization (initialized as None)
        self.average = None
        self.std = None

    def _normalize(self, data):
        """
        Normalize input data using loaded statistics.
        Matches the logic in data_loader.py
        """
        if self.average is None or self.std is None:
            print("Warning: Normalization statistics not loaded. Using raw data.")
            return data
            
        n = data.shape[0]
        t = data.shape[1]
        
        # Reshape logic mirrors data_loader.py but we need to be careful with dimensions
        # Input X is (N, T, C, F) or (N, T, C)
        # data_loader calculates mean/std on reshaped (N*T, -1)
        
        # For prediction, we just apply the formula broadcasting over N and T
        # average and std shapes are likely (1, C*F) or similar from training
        
        # Let's check how they were saved. 
        # In data_loader: average = np.mean(data_reshaped, axis=0, keepdims=True)
        # So shape is (1, Features_Flattened)
        
        original_shape = data.shape
        data_reshaped = data.reshape((n * t, -1))
        
        combine_max = self.average + 4 * self.std
        combine_min = self.average - 4 * self.std
        
        denominator = combine_max - combine_min
        denominator[denominator == 0] = 1e-8
        
        norm_data = 2 * (data_reshaped - combine_min) / denominator - 1
        return norm_data.reshape(original_shape)

    def _denormalize(self, data):
        """
        Denormalize output data to original scale.
        """
        if self.average is None or self.std is None:
            return data

        # Output data shape (N, T, C) usually
        # Our average/std might include all features (including index 4 which is target?)
        # Wait, in NeuroForcastDataset:
        # data = data[:, :, 0] if not use_graph (taking first feature)
        # But normalization happens BEFORE this selection in the dataset class.
        # So 'average' and 'std' stored in stats.npz correspond to ALL features (e.g. 5 features).
        
        # The model output corresponds to the Target (which is usually the first feature, index 0).
        # So we need to select the mean/std for the first feature to denormalize the output.
        
        # average shape from file: (1, Channel * Feature) if flattened?
        # Let's look at data_loader again.
        # data_reshaped = data.reshape((n*t, -1)). 
        # -1 collapses Channel and Feature dimensions.
        # So average is (1, C*F).
        
        # We need to reshape average back to (1, 1, C, F) to extract what we need.
        # self.input_size is #Channels.
        # Features? In demo X shape is (..., 5) or similar? 
        # Beignet: 89 channels. Affi: 239.
        # Let's infer feature dim from the size of average.
        
        if self.average.size % self.input_size == 0:
            num_features = self.average.size // self.input_size
        else:
            # Fallback
            num_features = 1
            
        # Reshape stats to (1, 1, C, F)
        avg_reshaped = self.average.reshape(1, 1, self.input_size, num_features)
        std_reshaped = self.std.reshape(1, 1, self.input_size, num_features)
        
        # Target is the first feature (index 0)
        # So we take mean/std for that feature only
        avg_target = avg_reshaped[:, :, :, 0] # (1, 1, C)
        std_target = std_reshaped[:, :, :, 0] # (1, 1, C)
        
        # Match output shape (N, T, C)
        # Broadcast will handle N and T
        
        combine_max = avg_target + 4 * std_target
        combine_min = avg_target - 4 * std_target
        
        denominator = combine_max - combine_min
        
        # Formula: norm = 2*(x - min)/denom - 1
        # x = (norm + 1) * denom / 2 + min
        
        denorm_data = (data + 1) * denominator / 2 + combine_min
        
        return denorm_data

    def predict(self, X):
        """
        Predict probability of neural activity.
        Args:
           X: Numpy array of shape (Sample_size, 20, Channel, Feature).
        Returns:
           Numpy array of shape (Sample_size, 20, Channel).
        """
        self.model.eval()
        
        # Preprocessing:
        # The demo extracts the first feature if use_graph=False (which seems to be the baseline)
        # X shape: (B, T, C, F). We need (B, T, C) or similar for the GRU input_size=C.
        # Let's assume we take the first feature as input, similar to demo: data[:, :, 0]
        # X is numpy array.
        
        # Extract meaningful input (first 10 steps usually, but model expects full sequence?)
        # Demo "predict" logic isn't explicitly shown as "take 10, predict 10" inside the model class 
        # but rather it processes the whole sequence.
        # However, for submission "predict" method:
        # "Only the first 10 steps have meaningful value. The last 10 steps are masked."
        # The model needs to return shape (Sample_size, 20, Channel).
        
        # Standardize input processing:
        # Take (B, 20, C, F) -> (B, 20, C) (using 1st feature) to match demo input_size=num_channels
        
        if X.ndim == 4:
            input_tensor = torch.from_numpy(X[:, :, :, 0]).float() # shape (B, 20, C)
        else:
            # Fallback if already reduced
            input_tensor = torch.from_numpy(X).float()
            
        input_tensor = input_tensor.to(self.device)
        
        # Normalize Input
        # Note: X is numpy. We should normalize BEFORE converting to tensor if we use numpy stats.
        # Let's revert the tensor conversion slightly or do it on numpy X.
        # But wait, self._normalize expects numpy array.
        
        X_norm = self._normalize(X)
        
        if X_norm.ndim == 4:
            input_tensor = torch.from_numpy(X_norm[:, :, :, 0]).float()
        else:
            input_tensor = torch.from_numpy(X_norm).float()
            
        input_tensor = input_tensor.to(self.device)

        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor) # shapes (B, 20, C)
            
        output_numpy = output_tensor.cpu().numpy()
        
        # Denormalize Output
        prediction = self._denormalize(output_numpy)
        
        return prediction

    def load(self):
        """
        Load the pre-trained model weights and normalization stats.
        """
        filename = "model.pth"
        stats_filename = "stats.npz"
        
        if self.monkey_name == 'beignet':
            filename = "model_beignet.pth"
            stats_filename = "stats_beignet.npz"
        elif self.monkey_name == 'affi':
            filename = "model_affi.pth"
            stats_filename = "stats_affi.npz"

        
        # In submission environment, weights are in the same dir as model.py
        # But locally they are in 'weights/' folder.
        # We try to handle both cases for convenience.
        
        # 1. Try local weights folder first (Development)
        weight_path_local = os.path.join(os.path.dirname(__file__), 'weights', filename)
        
        # 2. Try same directory (Submission / Codabench)
        weight_path_submission = os.path.join(os.path.dirname(__file__), filename)
        
        if os.path.exists(weight_path_local):
            weight_path = weight_path_local
        elif os.path.exists(weight_path_submission):
            weight_path = weight_path_submission
        else:
            weight_path = os.path.join('weights', filename)

        # Resolve paths for Stats
        stats_path_local = os.path.join(os.path.dirname(__file__), 'weights', stats_filename)
        stats_path_submission = os.path.join(os.path.dirname(__file__), stats_filename)
        
        if os.path.exists(stats_path_local):
            stats_path = stats_path_local
        elif os.path.exists(stats_path_submission):
            stats_path = stats_path_submission
        else:
            stats_path = os.path.join('weights', stats_filename)

        # Load Weights
        if not os.path.exists(weight_path):
             # print(f"Warning: Weight file not found at {weight_path}")
             pass
        else:
            try:
                state_dict = torch.load(weight_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                print(f"Successfully loaded model from {weight_path}")
            except Exception as e:
                print(f"Error loading model: {e}")

        # Load Stats
        if os.path.exists(stats_path):
            try:
                stats = np.load(stats_path)
                self.average = stats['average']
                self.std = stats['std']
                print(f"Successfully loaded stats from {stats_path}")
            except Exception as e:
                print(f"Error loading stats: {e}")
        else:
            print(f"Warning: Stats file not found at {stats_path}. Normalization will be disabled.")

if __name__ == "__main__":
    # Self-test code
    print("--- Running Model Self-Test ---")
    
    subject = 'beignet' # Change to 'affi' if needed
    print(f"Initializing Model for {subject}...")
    try:
        m = Model(subject)
        
        print("Testing Load...")
        m.load()
        
        print("Testing Predict...")
        # Create dummy input: (Batch=1, Time=20, Channel=89/239, Feature=5)
        # Note: Input size depends on subject
        channels = 89 if subject == 'beignet' else 239
        dummy_input = np.random.rand(2, 20, channels, 5)
        
        output = m.predict(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        assert output.shape == (2, 20, channels)
        print("Test PASSED: Output shape is correct.")
        
    except Exception as e:
        print(f"Test FAILED: {e}")
