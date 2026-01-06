#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "=========================================="
echo "Starting training for Beignet (Monkey 1)"
echo "Algorithm: DLinear + GNN"
echo "=========================================="
# Train Beignet (Best: alpha=1.0, dropout=0.0, hidden_dim=128)
python train.py --dataset beignet --model dlinear_gnn --epochs 150 --alpha 1.0 --dropout 0.0 --hidden_dim 128 --full_data --name "beignet_final_full"

echo ""
echo "=========================================="
echo "Starting training for Affi (Monkey 2)"
echo "Algorithm: AMAG"
echo "=========================================="
# Train Affi (Best: alpha=1.0, dropout=0.1, hidden_dim=128)
python train.py --dataset affi --model amag --epochs 150 --alpha 1.0 --dropout 0.1 --hidden_dim 128 --full_data --name "affi_final_full"

echo ""
echo "✅ All training completed successfully!"
