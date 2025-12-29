import torch
import torch.nn as nn
import os
import numpy as np

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """
    DLinear optimization for Neural Forecasting.
    Input: (B, 20, C) -> But we only use first 10.
    Output: (B, 20, C) -> But we only predict last 10 (first 10 are zeros).
    """
    def __init__(self, input_size, seq_len=10, pred_len=10, individual=False):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 3
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = individual
        self.channels = input_size
        
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x shape: (Batch, 20, C)
        # Slice the real input (first 10 steps)
        x_in = x[:, :self.seq_len, :] # (B, 10, C)
        
        # Decompose
        seasonal_init, trend_init = self.decompsition(x_in)
        # seasonal_init: (B, 10, C), trend_init: (B, 10, C)
        
        # Permute to (B, C, L) for Linear Layer
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x_out = seasonal_output + trend_output
        
        # Permute back to (B, L, C) -> (B, 10, C)
        x_out = x_out.permute(0, 2, 1)
        
        # Construct full 20-step output
        output = torch.cat([x[:, :self.seq_len, :], x_out], dim=1) # (B, 20, C)
        
        return output

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
        # Use DLinear instead of NFBaseModel
        self.model = DLinear(input_size=self.input_size)
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
        
        # Reshape logic mirrors data_loader.py
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
        # infer feature dim
        if self.average.size % self.input_size == 0:
            num_features = self.average.size // self.input_size
        else:
            num_features = 1
            
        # Reshape stats to (1, 1, C, F)
        avg_reshaped = self.average.reshape(1, 1, self.input_size, num_features)
        std_reshaped = self.std.reshape(1, 1, self.input_size, num_features)
        
        # Target is the first feature (index 0)
        avg_target = avg_reshaped[:, :, :, 0] # (1, 1, C)
        std_target = std_reshaped[:, :, :, 0] # (1, 1, C)
        
        combine_max = avg_target + 4 * std_target
        combine_min = avg_target - 4 * std_target
        
        denominator = combine_max - combine_min
        
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
        # X shape: (B, T, C, F). 
        # Extract meaningful input based on dataset logic (first feature only for now)
        
        # Normalize Input
        X_norm = self._normalize(X)
        
        if X_norm.ndim == 4:
            input_tensor = torch.from_numpy(X_norm[:, :, :, 0]).float() # shape (B, 20, C)
        else:
            # Fallback if already reduced
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
