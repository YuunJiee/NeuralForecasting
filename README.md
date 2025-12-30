# Neural Signal Forecasting Project

This repository contains the optimized implementation of the Neural Signal Forecasting pipeline. We have moved beyond the simple GRU baseline to dataset-specific architectures that maximize performance.

---

## 🏆 Current Performance & Architecture

We adopt a **Structure-based Strategy**, selecting the best model architecture for each subject's unique neural dynamics.

| Dataset | Best Algorithm | R2 Score | Characteristics |
| :--- | :--- | :--- | :--- |
| **Beignet** | **DLinear + GNN** | **~0.841** | High non-stationary drift. `DLinear` handles trend; `GNN` handles spatial correlation. |
| **Affi** | **AMAG (GRU + GNN)** | **~0.852** | Complex non-linear dynamics. `GRU` captures temporal patterns; `GNN` handles connectivity. |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n neural_forecasting python=3.10 -y
conda activate neural_forecasting
pip install -r requirements.txt
```

### 2. Training (Reproducing Results)

You can train the specific models using the `--model` argument.

**Train Beignet (DLinear + GNN):**
```bash
python train.py --dataset beignet --model dlinear_gnn --epochs 300
```
*   *Note: Uses LR=1e-3 for optimal convergence.*

**Train Affi (AMAG):**
```bash
python train.py --dataset affi --model amag --epochs 300
```

### 3. Evaluation

The `test_predict.py` script and `model.py` are now "smart". They will automatically select the best architecture based on the dataset name.

```bash
# Evaluate Beignet (Auto-selects DLinear+GNN)
python test_predict.py --dataset beignet

# Evaluate Affi (Auto-selects AMAG)
python test_predict.py --dataset affi
```

---

## 📂 Submission Guide

For the competition/leaderboard submission, you only need to upload the following files. The logic in `model.py` is self-contained and handles model selection/weight loading automatically.

### Required Files
1.  **`model.py`** (The inference logic)
2.  **`model_beignet_dlinear_gnn.pth`** (Beignet Weights)
3.  **`stats_beignet_dlinear_gnn.npz`** (Beignet Normalization Stats)
4.  **`model_affi.pth`** (Affi Weights)
5.  **`stats_affi.npz`** (Affi Normalization Stats)

> **Note**: `model.py` looks for weights in the same directory (for submission platform) OR in `weights/` directory (for local dev).

---

## 🛠 Project Structure

| File | Purpose |
| :--- | :--- |
| `model.py` | **Core Inference Script**. Contains `DLinearGNNModel` and `AMAGModel` classes. Submission entry point. |
| `train.py` | Training script. Supports `--model` argument to switch architectures. |
| `test_predict.py` | Evaluation script. Calculates R2/MSE on validation set. |
| `utils/trainer.py` | Contains `Trainer` and `AdvancedTrainer` classes. |
| `utils/graph_utils.py`| Computes Pearson Correlation for GNN initialization. |
| `weights/` | Directory storing trained model weights (`.pth`) and stats (`.npz`). |
| `RELAY_LOG.md` | Development log tracking experiments and decisions. |
