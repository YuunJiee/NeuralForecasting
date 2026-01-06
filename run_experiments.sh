#!/bin/bash
set -e

# Experiment: Grid Search Hyperparameters
# Alphas: 1.0, 5.0
# Dropouts: 0.0, 0.1
# Hidden Dims: 32, 64

alphas=(1.0 5.0)
dropouts=(0.0 0.1)
hidden_dims=(32 64)

echo "Starting Grid Search Experiments..."

for alpha in "${alphas[@]}"; do
  for dropout in "${dropouts[@]}"; do
    for h_dim in "${hidden_dims[@]}"; do
    
      # --- Beignet (DLinear + GNN) ---
      name="beignet_a${alpha}_d${dropout}_h${h_dim}"
      echo "----------------------------------------------------"
      echo "Running Experiment: $name"
      echo "alpha=$alpha, dropout=$dropout, hidden_dim=$h_dim"
      echo "----------------------------------------------------"
      
      # Use fewer epochs (100) for grid search to save time
      python train.py --dataset beignet --model dlinear_gnn --epochs 100 \
        --alpha $alpha --dropout $dropout --hidden_dim $h_dim --name "$name"
      
      
      # --- Affi (AMAG) ---
      name="affi_a${alpha}_d${dropout}_h${h_dim}"
      echo "----------------------------------------------------"
      echo "Running Experiment: $name"
      echo "alpha=$alpha, dropout=$dropout, hidden_dim=$h_dim"
      echo "----------------------------------------------------"
      
      python train.py --dataset affi --model amag --epochs 100 \
        --alpha $alpha --dropout $dropout --hidden_dim $h_dim --name "$name"
        
    done
  done
done

echo "✅ All grid search experiments completed!"
