import torch
import torch.nn as nn
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

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

        self.dropout = nn.Dropout(0.0) # Default dropout 0.0 for DLinear if not specified

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
            seasonal_output = self.dropout(self.Linear_Seasonal(seasonal_init))
            trend_output = self.dropout(self.Linear_Trend(trend_init))

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
        output, hidden = self.encoder(x)
        output = self.output_layer(output)
        return output

class AMAGModel(nn.Module):
    """
    AMAG-inspired Model:
    1. Temporal Encoding (GRU) per node (channel).
    2. Spatial Interaction (GNN) between nodes.
    3. Prediction readout.
    """
    def __init__(self, num_nodes, input_len=10, pred_len=10, hidden_dim=64, adj_init=None, dropout=0.0):
        super(AMAGModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # 1. Temporal Encoder (Shared across all nodes)
        # Input: (Batch, Time, 1) per node -> Output: (Batch, Hidden)
        # We process each node's time series independently first.
        self.temporal_encoder = nn.GRU(input_size=1, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        
        # 2. Spatial Interaction (GNN)
        # Learnable Adjacency Matrix
        if adj_init is not None:
            # register as buffer if we want it fixed, or parameter if learnable
            # Paper suggests dynamic, but here we initialize with Pearson and let it be learnable or static-ish
            # We'll make it a learnable weight initialized by Pearson
            self.adj = nn.Parameter(adj_init.clone())
        else:
            self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
            
        # GNN Layer (Simple Graph Conv W)
        self.gnn_fc = nn.Linear(hidden_dim, hidden_dim)
        self.gnn_fusion = nn.Linear(2 * hidden_dim, hidden_dim) # Fusion of Temporal + Spatial
        
        # 3. Readout
        self.readout = nn.Linear(hidden_dim, pred_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (Batch, InputLen, NumNodes) OR (Batch, TotalLen, NumNodes) if training
        # We only use the first input_len steps
        x_in = x[:, :self.input_len, :] # (Batch, 10, NumNodes)
        batch_size = x_in.size(0)
        
        # Reshape for Temporal Encoder: (Batch * NumNodes, InputLen, 1)
        # Permute to (Batch, NumNodes, InputLen) first
        x_node = x_in.permute(0, 2, 1).contiguous()
        x_flat = x_node.view(batch_size * self.num_nodes, self.input_len, 1)
        
        # Encode -> (Batch * NumNodes, InputLen, Hidden)
        out, h_n = self.temporal_encoder(x_flat)
        # Take last hidden state: (Batch * NumNodes, Hidden)
        temporal_feat = out[:, -1, :] 
        
        # Reshape back to (Batch, NumNodes, Hidden)
        temporal_feat = temporal_feat.view(batch_size, self.num_nodes, self.hidden_dim)
        
        # Spatial Interaction
        # A @ X @ W
        # A: (NumNodes, NumNodes)
        # X: (Batch, NumNodes, Hidden)
        adj_normalized = torch.softmax(self.adj, dim=1) 
        spatial_agg = torch.matmul(adj_normalized, temporal_feat) # (Batch, N, Hidden)
        
        # Transform 
        spatial_feat = self.dropout(torch.relu(self.gnn_fc(spatial_agg)))
        
        # Fusion
        combined = torch.cat([temporal_feat, spatial_feat], dim=-1)
        fused = self.dropout(torch.relu(self.gnn_fusion(combined))) # (Batch, N, Hidden)
        
        # Readout
        pred = self.readout(fused) # (Batch, N, PredLen)
        
        # Output format: (Batch, PredLen, NumNodes)
        pred = pred.permute(0, 2, 1)
        
        # Concatenate input (hist) + output (pred) to match old interface (B, 20, C)
        output = torch.cat([x_in, pred], dim=1)
        
        return output

 



class DLinearGNNModel(nn.Module):
    """
    Hybrid DLinear + GNN Model
    1. Decomposition (Trend + Seasonality)
    2. Linear Mapping (Time)
    3. GNN Interaction (Space) applied to both components
    4. Recomposition
    """
    def __init__(self, num_nodes, input_len=10, pred_len=10, adj_init=None, dropout=0.0, hidden_dim=64):
        super(DLinearGNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.pred_len = pred_len
        
        # Decomposition
        self.decomposition = SeriesDecomp(kernel_size=3)
        
        # Linear Mapping (Time) for Seasonal and Trend
        # Input: (Batch, Node, InputLen) -> Output: (Batch, Node, PredLen)
        # Treated as Linear Layer: (InputLen -> PredLen) shared or individual? 
        # Changed to MLP for Non-Linear Temporal Mapping
        mlp_dim = hidden_dim # Use passed hidden_dim instead of hardcoded 64
        self.linear_seasonal = nn.Sequential(
            nn.Linear(input_len, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, pred_len)
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(input_len, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, pred_len)
        )
        
        # GNN Interaction (Space)
        if adj_init is not None:
            self.adj = nn.Parameter(adj_init.clone())
        else:
            self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))
            
        # GNN weights for mixing spatial info
        # We process the PREDICTED future in spatial domain? Or the LATENT space?
        # Applying after Linear seems intuitive: refine the prediction spatially.
        self.gnn_seasonal = nn.Linear(pred_len, pred_len) 
        self.gnn_trend = nn.Linear(pred_len, pred_len)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (Batch, InputLen, NumNodes)
        x_in = x[:, :self.input_len, :]
        
        # Permute to (Batch, NumNodes, InputLen)
        x_in = x_in.permute(0, 2, 1)
        
        # Decompose
        seasonal_init, trend_init = self.decomposition(x_in)
        # Shapes: (Batch, NumNodes, InputLen)
        
        # Linear Mapping (Temporal)
        seasonal_output = self.linear_seasonal(seasonal_init) # (Batch, N, PredLen)
        trend_output = self.linear_trend(trend_init)          # (Batch, N, PredLen)
        
        seasonal_output = self.dropout(seasonal_output)
        trend_output = self.dropout(trend_output)
        
        # Spatial Interaction
        # A @ X
        adj_normalized = torch.softmax(self.adj, dim=1)
        
        # Seasonal GNN
        # A: (1, N, N) broadcast
        # X: (Batch, N, PredLen)
        seasonal_spatial = torch.matmul(adj_normalized, seasonal_output)
        seasonal_output = seasonal_output + self.dropout(torch.relu(self.gnn_seasonal(seasonal_spatial))) # Residual + Non-linear
        
        # Trend GNN
        trend_spatial = torch.matmul(adj_normalized, trend_output)
        trend_output = trend_output + self.dropout(torch.relu(self.gnn_trend(trend_spatial)))
        
        # Recompose
        x_out = seasonal_output + trend_output # (Batch, N, PredLen)
        
        # Output format
        x_out = x_out.permute(0, 2, 1) # (Batch, PredLen, N)
        
        output = torch.cat([x[:, :self.input_len, :], x_out], dim=1)
        return output


class Model:
    def __init__(self, monkey_name="", adj_init=None, model_type=None, dropout=0.0, hidden_dim=128, blur_sigma=0.0):
        """
        Initialize the model wrapper.
        Args:
            monkey_name (str): Name of the subject ('beignet' or 'affi').
            adj_init (torch.Tensor): Pearson correlation matrix for initialization.
            model_type (str): 'amag', 'dlinear_gnn', 'stndt', or 'dlinear_stndt'. 
                              If None, auto-selects best model for the subject.
        """
        self.monkey_name = monkey_name
        self.blur_sigma = blur_sigma
        
        # Auto-select best model if not specified
        if model_type is None:
            if self.monkey_name == 'beignet':
                self.model_type = 'dlinear_gnn'
            elif self.monkey_name == 'affi':
                self.model_type = 'amag'
            else:
                self.model_type = 'amag' # Fallback
        else:
            self.model_type = model_type
        
        # Configuration based on monkey_name
        if self.monkey_name == 'beignet':
            self.input_size = 89
        elif self.monkey_name == 'affi':
            self.input_size = 239
        else:
            if self.monkey_name == "":
                self.input_size = 89
            else:
                raise ValueError(f'No such a monkey: {self.monkey_name}')
        
        # Initialize the actual PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if adj_init is None:
             adj_init = torch.eye(self.input_size)
             
        if self.model_type == 'dlinear_gnn':
            print(f"Initializing DLinearGNNModel with dropout={dropout}, hidden_dim={hidden_dim}...")
            self.model = DLinearGNNModel(num_nodes=self.input_size, adj_init=adj_init, dropout=dropout, hidden_dim=hidden_dim)
        else:
            # Default to AMAG
            print(f"Initializing AMAGModel with hidden_dim={hidden_dim}...")
            self.model = AMAGModel(num_nodes=self.input_size, adj_init=adj_init, hidden_dim=hidden_dim)
            
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
        
        # Apply Gaussian Blur (Preprocessing)
        if self.blur_sigma > 0:
            # CRITICAL: Only blur the input history (first 10 steps) to avoid boundary artifacts
            # with the masked future (which is flat/repeated).
            input_steps = 10
            if X.shape[1] >= input_steps:
                 X[:, :input_steps] = gaussian_filter1d(X[:, :input_steps], sigma=self.blur_sigma, axis=1, mode='nearest')
            else:
                 X = gaussian_filter1d(X, sigma=self.blur_sigma, axis=1, mode='nearest')

        # Normalize Input
        X_norm = self._normalize(X)
        
        if X_norm.ndim == 4:
            input_tensor = torch.from_numpy(X_norm[:, :, :, 0]).float() # shape (B, 20, C)
        else:
            # Fallback if already reduced
            input_tensor = torch.from_numpy(X_norm).float()
            
        # Slice first 10 for input
        input_tensor_slice = input_tensor[:, :10, :]

        input_tensor_slice = input_tensor_slice.to(self.device)
        
        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor_slice) # shapes (B, 20, C)
            
        output_numpy = output_tensor.cpu().numpy()
        
        # Denormalize Output
        prediction = self._denormalize(output_numpy)
        
        # --- Calibration (Based on Leaderboard Feedback) ---
        # Affi has a systematic negative bias (Pred < GT).
        # We apply a mean-shift calibration to center the predictions.
        if self.monkey_name == 'affi':
            # calibration_bias = 12.057
            # prediction = prediction + calibration_bias
            pass
            
        # --- Temporal Smoothing (EMA) ---
        # Based on post-processing experiments:
        # Beignet (Alpha=0.6) -> Improved MSE by ~1000
        # Affi (Alpha=0.9) -> Improved MSE by ~7
        if self.monkey_name == 'beignet':
            alpha = 0.6
        elif self.monkey_name == 'affi':
            alpha = 0.9
        else:
            alpha = 1.0 # No smoothing for others
            
        if alpha < 1.0:
            # prediction shape: (Batch, 20, C)
            # We smooth forecasting window (steps 10-19)
            # Initialize t=10
            for t in range(11, 20):
                prediction[:, t, :] = alpha * prediction[:, t, :] + (1 - alpha) * prediction[:, t-1, :]
            
        return prediction

    def load(self):
        """
        Load the pre-trained model weights and normalization stats.
        """
        if self.model_type == 'amag':
            filename = f"model_{self.monkey_name}.pth"
            stats_filename = f"stats_{self.monkey_name}.npz"
        else:
            filename = f"model_{self.monkey_name}_{self.model_type}.pth"
            stats_filename = f"stats_{self.monkey_name}_{self.model_type}.npz"
        
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
        # Create dummy input: (Batch=1, Time=20, Channel=89/239, Feature=9)
        # Note: Input size depends on subject. Stats use 9 features.
        channels = 89 if subject == 'beignet' else 239
        dummy_input = np.random.rand(2, 20, channels, 9)
        
        output = m.predict(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        
        assert output.shape == (2, 20, channels)
        print("Test PASSED: Output shape is correct.")
        
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
