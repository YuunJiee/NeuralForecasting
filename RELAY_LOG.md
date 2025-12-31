# Relay Handoff Log - Day 1

**Author**: Yuunjiee
**Date**: 2025-12-29
**Current Best Configuration**: Pure DLinear + Global Normalization.

## Current Performance
- **Beignet**: Val Loss `0.01905` (Normalized) -> Est. MSE **80,000**.
- **Affi**: Val Loss `0.01863` (Normalized) -> Est. MSE **53,000**.
- **Total Est. MSE**: **133,000**.
- **Rank**: ~7 (Top 10).

---

# Relay Handoff Log - Day 2

**Author**: Yuunjiee
**Date**: 2025-12-30

## Final Logic & Strategy
We moved from a "One Model Fits All" approach to a **Structure-based Strategy**:
1.  **Beignet** (Strong Trend/Drift): Uses **DLinear + GNN**.
    *   DLinear handles the non-stationary drift.
    *   GNN refines spatial correlations.
    *   **Performance**: R2 ~0.841.
2.  **Affi** (Complex Dynamics): Uses **AMAG (GRU + GNN)**.
    *   GRU captures non-linear temporal dynamics better than Linear.
    *   **Performance**: R2 ~0.852 (Before Calibration).

## Experiments Summary
*   **DLinear (Pure)**: Good baseline, but missed spatial interaction.
*   **AMAG (GRU+GNN)**: Excellent for Affi, slightly weaker on Beignet trends.
*   **STNDT (Transformer/BERT)**: Explored. Good (R2 ~0.83) but didn't beat the specialized DLinear/AMAG models on this dataset size.
*   **Hybrid (DLinear+BERT)**: Over-engineered, performance dropped.

## Repository Cleanup
*   **Kept**: `AMAGModel`, `DLinearGNNModel` (and dependencies).
*   **Removed**: Experimental `STNDTModel` and `DLinearSTNDTModel` to keep `model.py` clean for submission.
*   **Weights**:
    *   `weights/model_beignet_dlinear_gnn.pth`
    *   `weights/stats_beignet_dlinear_gnn.npz`
    *   `weights/model_affi.pth`
    *   `weights/stats_affi.npz`

---

# Relay Handoff Log - Day 2 (Part 2: The Calibration Victory)

**Author**: Yuunjiee (with AI Advisor)
**Date**: 2025-12-30 (Pro)

## Critical Finding: Affi Bias & Post-Processing
We followed the "Inference-only" optimization strategy to squeeze out performance without re-training.

### 1. Affi Calibration (Bias Correction)
*   **Issue**: Systematic negative bias (-12.057) detected.
*   **Fix**: Added `prediction += 12.057` in `model.py`.
*   **Result**: Bias ~0. MSE reduced from 51,261 -> **51,110**.

### 2. Temporal Smoothing (EMA)
*   **Issue**: High frequency noise in predictions.
*   **Fix**: Applied Exponential Moving Average (EMA) to the prediction window (steps 10-19).
    *   **Beignet**: Strong drift needs smoothing. Used `Alpha=0.6`. **MSE reduced by ~1000 (70.5k -> 69.6k)**.
    *   **Affi**: Complex dynamics need less smoothing. Used `Alpha=0.9`. Slight stability improvement.

## Final Validated Scores
| Dataset | Strategy | Final MSE | R2 Score |
| :--- | :--- | :--- | :--- |
| **Beignet** | DLinear+GNN + EMA(0.6) | **69,605** | **0.8435** |
| **Affi** | AMAG + Bias(+12) + EMA(0.9) | **51,108** | **0.8422** |

## Next Person Action Items
*   **Do NOT Re-train**. The current weights + `model.py` logic are optimal.
*   **Submission**: Submit `model.py` and the 4 weight files.
