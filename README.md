# Neural Forecasting Project

This repository contains the implementation of a Neural Signal Forecasting model (GRU-based Baseline) for the competition.

## 📂 Project Structure

## 📂 Project Structure & File Purpose

**Where should I put my code?**

| Path | Purpose |
| :--- | :--- |
| **`models/`** | **Model Definitions** |
| `models/temporal.py` | **(Role A)** Time-series encoder (GRU, Transformer). Process `(Batch, Time, Channel)` data. |
| `models/spatial.py` | **(Role B)** Spatial encoder (GNN, CNN). Process brain topology & connectivity. |
| `models/hybrid_model.py` | **(Role D)** The "Motherboard" that connects Temporal and Spatial modules together. |
| **`notebooks/`** | **EDA & Experiments** |
| `notebooks/*.ipynb` | **(All Roles)** Place all Exploratory Data Analysis here (e.g., FFT, Spectrograms). |
| **`utils/`** | **Helper Functions** |
| `utils/data_loader.py` | Data loading, splitting, and `NeuroForcastDataset` class. |
| `utils/trainer.py` | **(Role C)** The training loop, validation logic, and loss calculation. |
| **`Root`** | **Execution & Submission** |
| `model.py` | **[CRITICAL]** The submission wrapper. Must import from `models/` and load weights. |
| `train.py` | Main script to run training (generates `weights/`). |
| `test_predict.py` | Script to evaluate model performance (MSE/R2). |


## 🚀 Quick Start

### 1. Environment Setup

Create the conda environment and install dependencies:

```bash
conda create -n neural_forecasting python=3.10 -y
conda activate neural_forecasting
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure the following data files are in the `data/` directory:
- `train_data_beignet.npz`
- `train_data_affi.npz`

### 3. Training

Train the model for a specific subject (dataset):

```bash
# Train Monkey 'beignet'
python train.py --dataset beignet

# Train Monkey 'affi'
python train.py --dataset affi
```

This will generate:
- `weights/model_{dataset}.pth`: Trained weights.
- `weights/stats_{dataset}.npz`: Normalization statistics (Important for correct inference!).

### 4. Evaluation

Run the evaluation script to check MSE and R2 scores on the validation set:

```bash
python test_predict.py
```

## 📊 Current Baseline Performance (GRU)

| Dataset | Split | MSE | R2 Score | Note |
|:---:|:---:|:---:|:---:|:---:|
| **Beignet** | Val (Forecasting) | ~110,951 | **0.7506** | Baseline (Normalized Input/Output) |
| **Affi** | Val (Forecasting) | ~59,784 | **0.8154** | Baseline (Normalized Input/Output) |

> **Note:** The high MSE is expected as the model output is denormalized to the original biological signal scale. The high R2 score indicates good trend prediction.

## 📝 Submission Guidelines

To submit to the competition platform, compress the following into a `zip` file:

1. `model.py`
2. `weights/model_beignet.pth`
3. `weights/stats_beignet.npz`
4. `weights/model_affi.pth`
5. `weights/stats_affi.npz`

**Ensure all files are either in the root of the zip or `weights/` is preserved as a subfolder.**

## 🛠 Development Roadmap

- [x] **Phase 1: Baseline** - Implement GRU model with correct normalization pipeline.
- [ ] **Phase 2: Feature Expansion** - Use all 5 input features (currently using only 1).
- [ ] **Phase 3: Architecture** - Implement Transformer / GNN for better spatial-temporal modeling.

## 👥 Team Roles & Responsibilities

| Role | Module | Key Responsibilities | Deliverables |
| :--- | :--- | :--- | :--- |
| **A (Temporal Lead)** | `models/temporal.py` | Time-series encoding (GRU/Transformer), prediction decoding. | `TemporalBlock` (nn.Module), Feature Embeddings |
| **B (Spatial Lead)** | `models/spatial.py` | Spatial analysis (EDA), GNN layers, adjacency matrix optimization. | `SpatialBlock` (nn.Module), Adjacency Matrix |
| **C (Training Lead)** | `utils/trainer.py` | Spectral analysis (EDA), Normalization strategies, Loss function design. | Optimized Hyperparameters (LR, Loss), `stats.npz` |
| **D (Integration)** | `model.py` | Module integration, Path management, Submission environment compliance. | Valid `submission.zip`, Integrated `Model` class |
