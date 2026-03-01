# Recruit Restaurant Visitor Forecasting

- URL: https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting
- Evaluation Metric: **RMSLE** (Root Mean Squared Logarithmic Error)
- Participants: 2,216 teams

## Problem Description

Predict the number of future visitors to restaurants in Japan. Data comes from two reservation systems:
- **Hot Pepper Gourmet (HPG)**: Similar to Yelp
- **AirREGI/Restaurant Board (air)**: Similar to Square POS

**Why RMSLE?**
- Penalizes underprediction more than overprediction
- Restaurants prefer overpreparation (extra inventory) over underpreparation (running out of food)
- Handles large value discrepancies better than RMSE

## Data Description

| File | Description |
|------|-------------|
| air_visit_data.csv | Historical visit data for air restaurants |
| air_reserve.csv | Reservation data for air restaurants |
| hpg_reserve.csv | Reservation data for HPG restaurants |
| air_store_info.csv | Store information (genre, area, location) |
| hpg_store_info.csv | HPG store information |
| store_id_relation.csv | Mapping between air and HPG store IDs |
| date_info.csv | Holiday information |
| sample_submission.csv | Submission format |

## Top Solutions Analysis

### 8th Place Solution (MaxHalford)
- **Model**: LightGBM with cross-validation
- **Key**: Targeted feature engineering
- **Score**: Public 0.468 / Private 0.509
- [GitHub](https://github.com/MaxHalford/kaggle-recruit-restaurant)

### 11th Place Solution (XIUQI1994)
- **Model**: Facebook Prophet
- **Key**: Rolling weighted mean + Weather data integration
- **Features**: Weather stations within 20km of each restaurant
- [GitHub](https://github.com/XIUQI1994/Kaggle_Recruit-Restaurant-Visitor-Forecasting_)

### 3rd Place (Kaggle Days)
- **Models**: LightGBM (0.509) + Keras (0.513) ensemble
- **Final Score**: 0.505
- **Key**: Weighted average of multiple models
- [GitHub](https://github.com/dkivaranovic/kaggledays-recruit)

## Key Feature Engineering Techniques

### 1. Rolling Statistics
```python
# Window sizes used by top solutions
windows = [7, 14, 21, 35, 63, 140, 280, 350, 420]

# Statistics computed
- mean, median, std, min, max
- exponentially weighted mean
```

### 2. Time-based Features
- Day of week, month, year
- Holiday flags (including Japan's Golden Week)
- Day before holiday marked as Friday

### 3. Lag Features
- Visitors from same day previous weeks
- Moving averages over different windows

### 4. External Data
- Weather data (rain, temperature)
- Station selection: within 20km with good coverage

### 5. Categorical Encoding
- Genre (Japanese, Italian, etc.)
- Area (Tokyo, Osaka, etc.)
- One-hot and label encoding

## Common Problems and Solutions

| Problem | Solution |
|---------|----------|
| Missing days (no visitors) | Resample to fill with 0 visitors |
| Outliers | Define as values outside 2.4σ confidence interval |
| New restaurants | Use genre/area aggregations |
| Time series leakage | Use rolling validation, not random split |

## Model Comparison

| Model | Strengths | Weaknesses | Typical Score |
|-------|-----------|------------|---------------|
| LightGBM | Fast, handles categorical well | Requires feature engineering | 0.50-0.52 |
| XGBoost | Robust, good generalization | Slower than LightGBM | 0.51-0.53 |
| Random Forest | Handles outliers well | Slower, memory intensive | 0.52-0.55 |
| Prophet | Built-in seasonality | Limited feature integration | 0.51-0.54 |
| Neural Network | Can learn complex patterns | Needs more data, tuning | 0.51-0.55 |

## Notebooks

1. `01_eda.ipynb` - Exploratory Data Analysis
2. `02_feature_engineering.ipynb` - Feature Engineering Study
3. `03_model_comparison.ipynb` - RF vs LightGBM vs XGBoost

## Key Learnings

1. **Feature engineering > Model selection** for this competition
2. **Rolling statistics** are the most important features
3. **Time series validation** is critical (no random split)
4. **Ensemble methods** improve final score by ~0.01
5. **External weather data** provides small but consistent improvement

## Improvement Log

### 2026-03-01: RFモデル改良 - Step 1: 休店日の取り扱い検証

**対象ノートブック**: `05_submission_rf.ipynb`

**背景・動機**:
RFモデルのベースライン（検証RMSLE ≈ 0.49）を改良するにあたり、3つの論点を整理した。
まず論点1「休店日の取り扱い」をコードで実際に検証し、改善余地を把握する。

**論点一覧**:
1. 休店日の取り扱い確認 ← 今回実施
2. グリッドサーチとバリデーション（次回予定）
3. 生成AIが提案した改善方法（時間があれば）

**現状の仕組み（cell-7）**:
```
全店舗 × 全日付のgrid_df生成 → train_dfをleft merge → 休店日はvisitors=NaN
→ rolling/lag計算（NaNはスキップ） → 結果をall_dfにmerge
```

**検証で確認する項目（cell-8〜9に追加）**:
1. grid_dfで休店日の`visitors`がNaNになっているか
2. `rolling_mean`がNaNをスキップして計算されているか
3. 休店日翌日の`lag_1`がNaN（→ -999補完）になるか
4. all_dfに休店日の行が含まれないか
5. 全体のNaN件数サマリー

**想定される問題点**:
- `lag_1`: カレンダー日ベースなので前日休店なら常にNaN → -999補完はRFにとって極端な値
- `rolling(7)`: 7カレンダー日窓。週1定休の店舗は実質6日分
- NaN=-999補完 → 中央値補完の方が良い可能性あり

**変更内容**:
- cell-7の後にマークダウンセル（検証の説明）とコードセル（検証コード）を追加
- 既存のモデル学習・提出コードには変更なし

---

### 2026-03-01: RFモデル改良 - Step 2: グリッドサーチの実装

**対象ノートブック**: `05_submission_rf.ipynb`

**背景・動機**:
ベースラインのRFモデルはハードコードされたパラメータ（n_estimators=200, max_depth=15等）を使用。
GridSearchCVで最適なハイパーパラメータを探索し、検証RMSLEの改善を図る。

**実装方針**:
- `sklearn.model_selection.PredefinedSplit` で時系列分割を実現（ランダムCV分割ではない）
- 検証期間: 2017-03-12以降（ベースラインと同一）
- スコアリング: `neg_mean_squared_error`（log空間のMSE = RMSLE²に対応）
- `refit=False`: パラメータ探索のみ。再学習は後続セルで実施

**探索パラメータ（324組み合わせ）**:

| パラメータ | ベースライン | 探索範囲 |
|-----------|------------|---------|
| n_estimators | 200 | [100, 200, 500] |
| max_depth | 15 | [10, 15, 20, None] |
| min_samples_split | 10 | [5, 10, 20] |
| min_samples_leaf | 5 | [3, 5, 10] |
| max_features | 'sqrt' | ['sqrt', 'log2', 0.5] |

**変更内容**:
1. cell-14〜15に追加: GridSearchCV マークダウン説明 + 実装コード
2. cell-16を更新: ハードコードパラメータ → `**best_params` に変更
3. cell-20を更新: 最終モデルも `**best_params` を使用

**ノートブックのセル構成（変更後）**:
```
cell-13: バリデーション分割（VALID_START）
cell-14: [NEW] GridSearchCV 説明（markdown）
cell-15: [NEW] GridSearchCV 実行 → best_params を出力
cell-16: [MODIFIED] best_params でRF学習・検証RMSLE表示
  ...
cell-20: [MODIFIED] best_params で全データ再学習（提出用）
```

**結果**: 実行後に best_params と RMSLE を記録する（TODO）

---

### 次回予定: Step 3 - 生成AI提案の改善方法

（ユーザーに確認が必要）

---

## References

- [8th Place Solution](https://github.com/MaxHalford/kaggle-recruit-restaurant)
- [11th Place Solution](https://github.com/XIUQI1994/Kaggle_Recruit-Restaurant-Visitor-Forecasting_)
- [Kaggle Days Solution](https://github.com/dkivaranovic/kaggledays-recruit)
- [Competition Page](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting)
