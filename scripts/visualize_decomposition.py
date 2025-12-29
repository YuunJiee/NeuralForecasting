import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_dataset
from model import DLinear

def main():
    dataset_name = 'beignet'
    data_filename = f'train_data_{dataset_name}.npz'
    input_dir = './data'
    
    # Load Data
    _, _, val_data = load_dataset(data_filename, input_dir=input_dir)
    
    # Select a sample with high activity
    # val_data: (N, 20, C, F)
    # Let's pick sample 0, Channel 0 (Band 0 usually dominant)
    sample_idx = 0
    channel_idx = 0
    
    # Prepare Input
    # Model expects (B, 20, C) tensor
    input_tensor = torch.from_numpy(val_data[sample_idx:sample_idx+1, :, :, 0]).float()
    
    # Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DLinear(input_size=89)
    model.load_state_dict(torch.load(f'weights/model_{dataset_name}.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Extract Components manually (since forward pass combines them)
    # We access the decomposition block directly
    with torch.no_grad():
        x_in = input_tensor[:, :10, :].to(device) # First 10 steps
        seasonal, trend = model.decompsition(x_in)
        
        # Get Outputs of the Linear Layers
        # We want to see what the model *predicts* for Trend vs Seasonal
        # Permute for Linear
        seasonal_p = seasonal.permute(0, 2, 1)
        trend_p = trend.permute(0, 2, 1)
        
        seasonal_out = model.Linear_Seasonal(seasonal_p)
        trend_out = model.Linear_Trend(trend_p)
        
        # Permute back
        seasonal_out = seasonal_out.permute(0, 2, 1)
        trend_out = trend_out.permute(0, 2, 1)
        
        # To Numpy
        x_in_np = x_in.cpu().numpy()[0, :, channel_idx]
        trend_np = trend.cpu().numpy()[0, :, channel_idx]
        seasonal_np = seasonal.cpu().numpy()[0, :, channel_idx]
        
        trend_out_np = trend_out.cpu().numpy()[0, :, channel_idx]
        seasonal_out_np = seasonal_out.cpu().numpy()[0, :, channel_idx]
        
        total_pred = trend_out_np + seasonal_out_np
        gt_future = val_data[sample_idx, 10:, channel_idx, 0]

    # Calculate Variance Ratio
    var_trend = np.var(trend_np)
    var_seasonal = np.var(seasonal_np)
    ratio = var_trend / (var_seasonal + 1e-9)
    
    print(f"Variance Trend: {var_trend:.4f}")
    print(f"Variance Seasonal: {var_seasonal:.4f}")
    print(f"Trend/Seasonal Ratio: {ratio:.2f}")

    # Plot
    plt.figure(figsize=(12, 6))
    
    # 1. Decomposition of History
    plt.subplot(2, 1, 1)
    plt.title(f"Decomposition of History (Steps 0-9) - Ratio: {ratio:.1f}x")
    plt.plot(x_in_np, 'k-', label='Input Raw', linewidth=2)
    plt.plot(trend_np, 'r--', label='Trend Component')
    plt.plot(seasonal_np, 'b:', label='Seasonal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Prediction Components
    plt.subplot(2, 1, 2)
    plt.title("Prediction Components (Steps 10-19)")
    plt.plot(gt_future, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(total_pred, 'g-', label='Model Prediction')
    plt.plot(trend_out_np, 'r--', label='Trend Prediction')
    plt.plot(seasonal_out_np, 'b:', label='Seasonal Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'results/figures/dlinear_decomposition.png'
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
