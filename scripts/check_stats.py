import numpy as np
import os

stats_path = 'weights/stats_beignet.npz'
if os.path.exists(stats_path):
    data = np.load(stats_path)
    print("Keys:", data.files)
    if 'average' in data:
        print("Average shape:", data['average'].shape)
    if 'std' in data:
        print("Std shape:", data['std'].shape)
else:
    print(f"File not found: {stats_path}")
