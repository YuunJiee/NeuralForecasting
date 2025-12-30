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
**Date**: 2025-12-30 (Afternoon)

## Critical Finding: Affi Bias
After analyzing the leaderboard ranking, we discovered `Affi` MSE was higher than expected despite high R2.
*   **Investigation**: `scripts/investigate_bias.py` revealed a systematic **negative bias of -12.057**. The model was consistently under-predicting the magnitude.
*   **Root Cause**: Likely due to the heavy `Dropout` or `Normalization` shift in the GRU during training not perfectly aligning with the validation distribution.

## Final Solution: Test-Time Calibration
We implemented a hard-coded calibration in `model.py`'s `predict()` function:
```python
if self.monkey_name == 'affi':
    prediction = prediction + 12.057
```
*   **Result**: Bias reduced to **~0.0**. MSE dropped from 51,261 to **51,115**.

## Next Person Action Items
*   **Do NOT Re-train Affi** unless necessary. The current weights + calibration are optimal.
*   **Beignet**: `DLinear` requires high LR (1e-3). If re-training, check `train.py`.
*   **Submission**: Submit `model.py` and the 4 weight files. The code automatically handles everything.
