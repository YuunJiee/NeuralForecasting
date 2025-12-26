# Neural Signal Forecasting Project (Baseline GRU)

This repository contains the full implementation of the Neural Signal Forecasting pipeline for the competition. It includes the baseline GRU model, a robust training framework, and a verification system for correct submission formatting.

---

## 👥 Team Roles & Responsibilities

| Role | Module | Key Responsibilities | Deliverables |
| :--- | :--- | :--- | :--- |
| **A (Temporal Lead)** | `models/temporal.py` | Time-series encoding (GRU/Transformer), prediction decoding. | `TemporalBlock` (nn.Module), Feature Embeddings |
| **B (Spatial Lead)** | `models/spatial.py` | Spatial analysis (EDA), GNN layers, adjacency matrix optimization. | `SpatialBlock` (nn.Module), Adjacency Matrix |
| **C (Training Lead)** | `utils/trainer.py` | Spectral analysis (EDA), Normalization strategies, Loss function design. | Optimized Hyperparameters (LR, Loss), `stats.npz` |
| **D (Integration)** | `model.py` | Module integration, Path management, Submission environment compliance. | Valid `submission.zip`, Integrated `Model` class |

---

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

---

## 🚀 Quick Start

### 1. Environment Setup

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

Train the model (Generates weights and normalization stats):

```bash
# Train Monkey 'beignet'
python train.py --dataset beignet

# Train Monkey 'affi'
python train.py --dataset affi
```

### 4. Evaluation

Verify performance on the validation set:

```bash
python test_predict.py
```

---

## 📊 Current Baseline Performance (GRU)

| Dataset | Split | MSE | R2 Score | Note |
|:---:|:---:|:---:|:---:|:---:|
| **Beignet** | Val (Forecasting) | ~110,951 | **0.7506** | Baseline (Normalized Input/Output) |
| **Affi** | Val (Forecasting) | ~59,784 | **0.8154** | Baseline (Normalized Input/Output) |

> **Note:** The high MSE is due to the denormalization of the output to the original biological signal scale. The high R2 score confirms that the model is correctly predicting the trends.

---

## 📝 Submission Guidelines

To submit to the competition platform, compress the following into a **single zip file**:

1. `model.py`
2. `weights/model_beignet.pth`
3. `weights/stats_beignet.npz`
4. `weights/model_affi.pth`
5. `weights/stats_affi.npz`

**Important:** The `model.py` submission file is pre-configured to automatically load the correct weights and normalization statistics for each monkey.

---

## 🛠 Development Roadmap

- [x] **Phase 1: Baseline** - Implement GRU model with correct normalization pipeline (Completed by Role C & D).
- [ ] **Phase 2: Feature Expansion** - Use all 5 input features (currently using only 1). **[High Impact]**
- [ ] **Phase 3: Architecture** - Implement Transformer / GNN for better spatial-temporal modeling.
