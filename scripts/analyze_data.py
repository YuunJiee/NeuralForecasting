import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_bands(filepath, dataset_name):
    print(f"\n--- Analyzing Bands for {dataset_name} ---")
    try:
        data = np.load(filepath)['arr_0'] # Shape (N, T, C, F)
        
        # Simulate Train Split (80%)
        train_end = int(len(data) * 0.8)
        train_data = data[:train_end] 
        # Shape: (Samples, Time, Channels, Bands=9)
        
        num_bands = train_data.shape[3]
        
        stats = []
        
        print(f"{'Band':<5} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10} | {'% < 0':<10}")
        print("-" * 70)
        
        for b in range(num_bands):
            band_data = train_data[:, :, :, b].flatten()
            
            _min = np.min(band_data)
            _max = np.max(band_data)
            _mean = np.mean(band_data)
            _std = np.std(band_data)
            _neg_ratio = np.mean(band_data < 0) * 100
            
            stats.append(band_data)
            print(f"{b:<5} | {_min:<10.2f} | {_max:<10.2f} | {_mean:<10.2f} | {_std:<10.2f} | {_neg_ratio:<9.2f}%")

        plt.figure(figsize=(12, 6))
        plt.boxplot(stats, labels=[f'B{i}' for i in range(num_bands)], showfliers=False)
        plt.title(f"{dataset_name} - Distribution per Frequency Band (No Outliers)")
        plt.xlabel("Frequency Band")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/figures/bands_distribution_{dataset_name}.png')
        print(f"Saved band plot to results/figures/bands_distribution_{dataset_name}.png")

        # Save to CSV
        csv_path = f'results/stats/band_stats_{dataset_name}.csv'
        os.makedirs('results/stats', exist_ok=True)
        with open(csv_path, 'w') as f:
            header = "Band,Min,Max,Mean,Std,NegPercentage\n"
            f.write(header)
            for i, data in enumerate(stats):
                _min = np.min(data)
                _max = np.max(data)
                _mean = np.mean(data)
                _std = np.std(data)
                _neg_ratio = np.mean(data < 0) * 100
                f.write(f"{i},{_min:.4f},{_max:.4f},{_mean:.4f},{_std:.4f},{_neg_ratio:.4f}\n")
        print(f"Saved stats to {csv_path}")

        # Generate Comprehensive Report
        report_path = f'results/stats/data_report_{dataset_name}.txt'
        flat_all = train_data.flatten()
        percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
        perc_vals = np.percentile(flat_all, percentiles)

        with open(report_path, 'w') as f:
            f.write(f"=== Dataset Information: {dataset_name} ===\n")
            f.write(f"Shape: {train_data.shape} (Samples, Timesteps, Channels, Bands)\n")
            f.write(f"Total Elements: {flat_all.size}\n")
            f.write(f"Data Type: {train_data.dtype}\n\n")
            
            f.write("=== Overall Statistics ===\n")
            f.write(f"Min Value: {np.min(flat_all):.4f}\n")
            f.write(f"Max Value: {np.max(flat_all):.4f}\n")
            f.write(f"Mean Value: {np.mean(flat_all):.4f}\n")
            f.write(f"Std Dev: {np.std(flat_all):.4f}\n\n")
            
            f.write("=== Distribution (Percentiles) ===\n")
            for p, v in zip(percentiles, perc_vals):
                f.write(f"{str(p):>5}% : {v:.4f}\n")
            f.write("\n")
            
            f.write("=== Band-wise Summary ===\n")
            f.write(f"{'Band':<5} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}\n")
            f.write("-" * 60 + "\n")
            for i, data in enumerate(stats):
                f.write(f"{i:<5} | {np.min(data):<10.2f} | {np.max(data):<10.2f} | {np.mean(data):<10.2f} | {np.std(data):<10.2f}\n")

        print(f"Saved comprehensive report to {report_path}")

    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")

def main():
    data_dir = '../data'
    files = {
        'beignet': 'train_data_beignet.npz',
        'affi': 'train_data_affi.npz'
    }
    
    os.makedirs('results/figures', exist_ok=True)
    
    for name, f in files.items():
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            analyze_bands(path, name)
        else:
            print(f"File not found: {path}")

if __name__ == "__main__":
    main()
