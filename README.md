## 🎯 任務目標

本競賽的核心挑戰在於**預測未來神經訊號（Neural Signal Forecasting）**。神經訊號具有高度的時間依賴性與空間相關性，且在不同錄製場次（Sessions）之間存在「跨日漂移（Day-to-day drift）」的現象。

- **預測任務**：給定過去 10 個時間步的訊號，預測未來 10 個時間步的活動。
    
- **輸入數據**：`numpy array` 形狀為 `(Sample_size, 20, Channel, Feature)`。
    
    - 前 10 步為真實觀察值，後 10 步為遮蓋後的重複值。
        
    - `Feature[0]` 為目標訊號，`Feature[1:]` 為不同頻段的分解訊號。
        
- **輸出要求**：`numpy array` 形狀為 `(Sample_size, 20, Channel)`。
    
    - 必須包含原始的前 10 步與預測的後 10 步。
        

## 📊 評分標準

1. **預測準確度**：主要以 $R^2$ 或相關係數衡量預測值與真實值（Ground Truth）的接近程度。
    
2. **泛化能力**：模型在「未見場次（Unseen Sessions）」上的表現最為關鍵。
    
3. **推論延遲**：低延遲模型在 BCI 應用中更具競爭力。
    

## 📚 核心文獻研讀建議

針對本次開發使用的 **TE-SI-TR** 框架，組員應針對負責模組研讀以下兩篇 NeurIPS 論文：

|   |   |   |
|---|---|---|
|**角色**|**優先研讀文獻**|**研讀重點**|
|**A｜Temporal Lead**|[**STNDT (2022)**](https://www.google.com/search?q=NeurIPS-2022-stndt-modeling-neural-population-activity-with-spatiotemporal-transformers-Paper-Conference.pdf "null")|Transformer 在神經訊號的 Embedding 方式、時間注意力機制。|
|**B｜Spatial Lead**|[**AMAG (2023)**](https://www.google.com/search?q=NeurIPS-2023-amag-additive-multiplicative-and-adaptive-graph-neural-network-for-forecasting-neuron-activity-Paper-Conference.pdf "null")|**TE-SI-TR 框架**、加法/乘法交互模組 (Add/Modulator)、鄰接矩陣初始化。|
|**C｜Training Lead**|[**STNDT (2022)**](https://www.google.com/search?q=NeurIPS-2022-stndt-modeling-neural-population-activity-with-spatiotemporal-transformers-Paper-Conference.pdf "null")|**Contrastive Learning Loss (對比學習損失)** 的實作細節。|
|**D｜Integration**|[**AMAG (2023)**](https://www.google.com/search?q=NeurIPS-2023-amag-additive-multiplicative-and-adaptive-graph-neural-network-for-forecasting-neuron-activity-Paper-Conference.pdf "null")|掌握模型整體的數據流向與預測 (Forecasting) 邏輯。|

## 🔍 EDA (探索性資料分析) 關鍵點

在開發模型前，Role B 與 Role C 必須先完成以下分析：

1. **空間相關性分析 (Spatial Correlation)**：計算神經元間的 Pearson 相關係數，作為 GNN 鄰接矩陣的初始化依據。
    
2. **頻譜特徵檢查 (Frequency Features)**：分析 `Feature[1:9]` 與 `Feature[0]` 的關聯性，確認頻譜分解是否包含預測關鍵資訊。
    
3. **跨日漂移分析 (Drift Analysis)**：比較不同 Session 的訊號分布，決定正規化策略（全域 vs. 樣本級）。
    

## 🏗️ 資料夾架構

```
/NeuralForecasting_Project
│
├── /data                 # 存放數據集 (.npz)
├── /notebooks            # Role B/C: EDA 分析與視覺化 (.ipynb)
├── /models               # 模型架構定義
│   ├── temporal.py       # Role A: GRU/Transformer 模組
│   ├── spatial.py        # Role B: GNN/Adjacency 模組
│   └── hybrid_model.py   # Role D: 整合後的 TE-SI-TR 類別
│
├── /utils                # 輔助工具
│   ├── data_loader.py    # 正規化、頻率特徵提取、Data Augmentation
│   └── trainer.py        # Role C: 訓練邏輯、Huber Loss、對比學習
│
├── /weights              # 存放訓練好的權重 (.pth)
├── train.py              # 主執行程式：讀取資料並啟動訓練
├── test_predict.py       # 本地評測：模擬官方 Codabench 評分邏輯
└── model.py              # 最終提交介面：包含 load() 與 predict()

```

## 👥 團隊分工表

|   |   |   |   |
|---|---|---|---|
|**角色**|**負責模組**|**每日關鍵任務**|**產出標準**|
|**A｜Temporal Lead**|`temporal.py`|負責時序編碼（GRU）與預測解碼。|輸出 $(N, T, C, H)$ 特徵|
|**B｜Spatial Lead**|`spatial.py`|**EDA 空間分析**、GNN 層、鄰接矩陣優化。|提供通道間特徵混合模組|
|**C｜Training Lead**|`trainer.py`|**EDA 頻譜分析**、正規化、Loss 函數設計。|最小化殘差與解決漂移問題|
|**D｜Integration**|`model.py`|模組串接、路徑管理、環境格式確保。|產出符合規範之 `submission.zip`|

## 📅 一週時間表 (每日提交策略)

_目標：每天 18:00 前產生一個可提交的版本並上傳 LB。_

- **Day 1: 數據洞察與基礎工程**
    
    - **B/C**: 完成 EDA (相關性矩陣、頻譜分析、漂移檢查)。
        
    - **A/D**: 轉化 Demo 為 Python 模組；D 建立 `model.py` 骨架。
        
    - **提交**：基於 EDA 正規化策略的基礎 GRU。
        
- **Day 2: 空間關係接入**
    
    - **B**: 根據 EDA 結果實作靜態圖卷積層。
        
    - **A**: 實作序列預測解碼邏輯（確保輸出 20 步）。
        
    - **提交**：GRU + GCN 結構。
        
- **Day 3: 核心架構整合 (TE-SI-TR)**
    
    - **D**: 整合 A (時序) + B (空間交互) + C (特徵特徵)。
        
    - **提交**：完整時空預測模型 V1。
        
- **Day 4: 抗噪聲強化**
    
    - **C**: 引入 Huber Loss 與對比學習；**B**: 測試自適應鄰接矩陣。
        
    - **提交**：強化版模型，觀察泛化能力變化。
        
- **Day 5: 漂移優化與集成**
    
    - **全員**: 調整超參數解決跨天數據偏移。
        
    - **D**: 執行 Seed Ensemble (集成學習)。
        
    - **提交**：最終候選模型。
        
- **Day 6: Final Tweak**
    
    - 最終代碼清洗與規格檢查。
        

## ⚠️ 重要提醒 (CAUTION)

1. **路徑問題**：`load()` 內加載權重務必使用 `os.path.join(os.path.dirname(__file__), 'model.pth')`。
    
2. **數據一致性**：預測時的正規化參數必須與訓練時儲存的 `.npz` 一致。
    
3. **ZIP 格式**：壓縮時直接選取檔案進行壓縮，不可包含外層資料夾。
