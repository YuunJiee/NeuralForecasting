import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_dataset, NeuroForcastDataset

def plot_processed_distribution(dataset_name):
    print(f"--- Plotting Processed Distribution: {dataset_name} ---")
    data_filename = f'train_data_{dataset_name}.npz'
    
    try:
        # Load Raw
        train_data, _, _ = load_dataset(data_filename, input_dir='./data')
        
        # Process via Dataset Class
        ds = NeuroForcastDataset(train_data, use_graph=False, target_channels=239)
        processed_data = ds.data # (N, T, C, F)
        
        # Flatten
        flat_data = processed_data.flatten()
        
        # Stats
        _min = np.min(flat_data)
        _max = np.max(flat_data)
        _mean = np.mean(flat_data)
        _std = np.std(flat_data)
        
        print(f"Stats - Min: {_min:.2f}, Max: {_max:.2f}, Mean: {_mean:.2f}, Std: {_std:.2f}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(flat_data, bins=100, density=True, alpha=0.7, color='green', label='Processed Data')
        
        plt.title(f"Processed Distribution ({dataset_name})\n(Clip -> Log1p -> RobustScale)")
        plt.xlabel("Value (Scaled)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add text box with stats
        stats_text = f"Min: {_min:.2f}\nMax: {_max:.2f}\nMean: {_mean:.2f}\nStd: {_std:.2f}"
        plt.text(0.95, 0.95, stats_text, 
                 transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        out_path = f'results/figures/processed_distribution_{dataset_name}.png'
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_processed_distribution('beignet')
