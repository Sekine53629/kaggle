# Presentation Materials — Procedure Guide

## General Workflow for All Competitions

This guide defines the mandatory order for creating presentation materials (`説明用資料`) in each competition. All competitions under `competitions/` MUST follow this workflow.

---

## 【MANDATORY RULE】Strict Adherence to Creation Order

**The following order MUST be followed. Do NOT proceed to the next step until the previous step is fully completed AND all verification gates are passed.**

```
01_EDA → 02_Target & Feature Design → 03_Model Design (3-1, 3-2, ...) → 04_Comparison → 05_Summary
```

---

### Step 1: 01_EDA Analysis

- **File**: `01_EDA分析_日本語版.ipynb`
- **Content**: Exploratory Data Analysis (data quality, temporal patterns, distributions, key findings)
- **Output**: Key findings list, feature candidates, validation strategy

#### Verification Gate (must pass before Step 2)

| # | Check | How to Verify |
|---|-------|---------------|
| 1 | All tables loaded without error | Cell runs with no exceptions |
| 2 | Join key coverage documented | Print overlap % for all join keys |
| 3 | Missing data classified | Structural vs. random gaps clearly separated |
| 4 | Target distribution analyzed | Histogram + stats printed; skewness noted |
| 5 | Temporal patterns identified | DOW/month effects quantified with plots |
| 6 | Holiday effects analyzed | Per-period visitor ratios computed |
| 7 | Train/test distribution compared | DOW, holiday, genre overlap checked |
| 8 | Feature candidates listed | At least 10 candidate features with rationale |
| 9 | Intermediate data saved | `01_eda_results.pkl` exists in `intermediate/` |

---

### Step 2: 02_Target Variable & Feature Design

- **File**: `02_目的変数と説明変数の設計_{date}.ipynb`
- **Content**:
  - Target variable definition and transformation rationale (e.g., log1p)
  - Feature classification and design policy
  - Feature creation with incremental effectiveness verification
  - Feature importance and correlation analysis
  - **Learning period optimization** (best training start date)
  - **NaN handling strategy comparison** (fill_zero, median, ffill, leave_nan)
  - **Rolling feature configuration test** (calendar vs business-day, window sizes, EWM)
- **Prerequisite**: Step 1 must be completed with all gates passed

#### Verification Gate (must pass before Step 3)

| # | Check | How to Verify |
|---|-------|---------------|
| 1 | All features created without error | Full notebook runs top-to-bottom |
| 2 | No feature leakage | Correlation check: no feature > 0.95 with target |
| 3 | Feature importance computed | LightGBM importance plot generated |
| 4 | Incremental score improvement shown | Each feature category shows delta RMSLE |
| 5 | Learning period decided | Multiple periods tested; best one selected with evidence |
| 6 | NaN strategy decided | Multiple strategies tested; best one selected with evidence |
| 7 | Rolling config decided | Multiple configs tested; best one selected with evidence |
| 8 | `confirmed_settings` dict populated | Contains: `best_train_start`, `best_nan_strategy`, `best_rolling_config` |
| 9 | Intermediate data saved | `02_feature_design.pkl` exists with all keys |
| 10 | Feature count documented | Final feature list printed with category breakdown |

#### `confirmed_settings` Required Keys

```python
confirmed_settings = {
    'best_train_period': str,           # 表示用ラベル（例: '直近15ヶ月'）
    'best_train_start': str,            # 03用の絶対日付（例: '2015-12-12'）
    'best_train_period_days': int,      # CV検証で使った相対日数（例: 456）
    'best_nan_strategy': str,           # 例: 'NaN(デフォルト)'
    'best_rolling_config': str,         # 例: '現行のみ'
}
```

**注意**: 03-xノートブックは上記のキー名を**正確に**使用すること。以下はNG:
- `TRAIN_START` → 正しくは `best_train_start`
- `nan_strategy` → 正しくは `best_nan_strategy`
- `rolling_config` → 正しくは `best_rolling_config`

---

### Step 3: 03_Data Model Design (Multiple Notebooks)

One notebook per model. **Do NOT start until Step 2 is fully completed.**

| No. | Filename Pattern | Content |
|-----|-----------------|---------|
| 3-1 | `03-1_{ModelName}モデル_{date}.ipynb` | First model (typically LightGBM as baseline) |
| 3-2 | `03-2_{ModelName}モデル_{date}.ipynb` | Second model (e.g., XGBoost) |
| 3-3 | `03-3_{ModelName}モデル_{date}.ipynb` | Third model (e.g., RandomForest) |
| 3-4 | `03-4_{ModelName}モデル_{date}.ipynb` | Fourth model (e.g., CatBoost) |

- **Common structure for each notebook**:
  1. Load `confirmed_settings` from 02 pickle; **キー名を正確に使用して** apply training period filter
  2. Model overview and selection rationale
  3. Default hyperparameter training and cross-validation **（02のval_foldsを使用）**
  4. Residual analysis
  5. **ルールベースベースライン**: store×DOW中央値のRMSLEを算出（改善幅の基準）
  6. **Optuna hyperparameter tuning** (CV平均で最適化。50 trials for GBDT, 30 for RF)
  7. Retrain with tuned parameters
  8. Prediction output and model save
- **Prerequisite**: Step 2 must be completed with finalized feature design and `confirmed_settings`
- **IMPORTANT**: Create in order 03-1 -> 03-2 -> 03-3 -> 03-4 (each includes comparison with prior models)
- **CRITICAL: CV戦略の統一**: 03-xのCVは02で設計した`val_folds`を使う。`TimeSeriesSplit(n_splits=3)`は使用禁止（02の5フォールドCVとスコアが比較不能になるため）

#### Verification Gate per Model (must pass before next model or Step 4)

| # | Check | How to Verify |
|---|-------|---------------|
| 1 | `confirmed_settings` loaded correctly | Print and verify all 5 keys match 02's output (`best_train_period`, `best_train_start`, `best_train_period_days`, `best_nan_strategy`, `best_rolling_config`) |
| 2 | Training data filtered by `best_train_start` | `train_data.visit_date.min()` >= `best_train_start` |
| 3 | NaN strategy applied per `best_nan_strategy` | Check NaN counts before/after; strategy matches (RF例外: NaN非対応のため常に-999) |
| 4 | **CV戦略が02と同一** | 02の`val_folds`を使用（`TimeSeriesSplit`は使用禁止） |
| 5 | CV completed without error | All folds produce valid scores |
| 6 | Default RMSLE documented | Score printed and stored in results dict |
| 7 | **Optunaがval_foldsベースのCV**で最適化 | objective関数がCV平均を返す（Single Split過適合を防止） |
| 8 | Tuned RMSLE <= Default RMSLE | Tuning should not degrade performance |
| 9 | **ルールベースベースラインとの比較** | store×DOW中央値のRMSLEを算出し、MLモデルの改善幅を明示 |
| 10 | Residual analysis done | Residual distribution and per-segment analysis plotted |
| 11 | Model file saved | `.joblib` / `.json` / `.cbm` file exists in `intermediate/` |
| 12 | Intermediate results saved | `03-x_{model}_results.pkl` exists with required keys |

#### Required Keys in 03-x Results Pickle

```python
results = {
    # 予測値（single split）
    'valid_pred': np.ndarray,       # 元スケールの予測値
    'valid_pred_log': np.ndarray,   # log1pスケールの予測値
    'valid_actual': np.ndarray,     # 元スケールの実測値
    'residuals': np.ndarray,        # log1p空間の残差
    # スコア
    'score_single': float,          # Single split RMSLE
    'cv_scores': list,              # val_foldsベースのCV（02と同一フォールド）
    'cv_mean': float,
    'cv_std': float,
    # モデル情報
    'params': dict,                 # デフォルトパラメータ
    'best_iteration': int,
    'feature_importance': pd.DataFrame,
    # Optunaチューニング結果
    'tuned_params': dict,
    'tuned_score': float,
    'tuned_pred': np.ndarray,
    'tuned_pred_log': np.ndarray,
    'optuna_best_params': dict,
    'optuna_best_value': float,
}
```

#### confirmed_settings 読み込みテンプレート（03-x共通）

```python
# 正しい読み込みパターン
confirmed = prev_02['confirmed_settings']
print('=== confirmed_settings ===')
for k, v in confirmed.items():
    print(f'  {k}: {v}')

# キー存在チェック（サイレント失敗を防止）
required_keys = ['best_train_start', 'best_train_period_days',
                 'best_nan_strategy', 'best_rolling_config']
for key in required_keys:
    assert key in confirmed, f'Missing key: {key}'

# 学習データフィルタ
train_df = train_df[train_df['visit_date'] >= confirmed['best_train_start']].reset_index(drop=True)

# CV用のval_folds読み込み
val_folds = prev_02['val_folds']
```

---

### Step 4: 04_Model Performance Comparison & Submission

- **File**: `04_各モデルの性能比較_{date}.ipynb`
- **Content**:
  - Cross-model performance summary table (default vs. tuned)
  - Prediction correlation and scatter plots across models
  - Segment-level error analysis (DOW, visitor range, genre)
  - Feature importance cross-comparison (normalized, rank correlation)
  - Ablation study (feature category removal impact)
  - Ensemble strategy evaluation (simple average + Optuna weighted)
  - **ルールベース予測（store×DOW中央値）とのアンサンブル**
  - **テストデータへの予測パイプライン実装** (`sample_submission.csv` の `air_store_id_YYYY-MM-DD` 形式に対応)
  - **最終submission.csvの生成と提出**
- **Prerequisite**: ALL notebooks in Step 3 must be completed
- **IMPORTANT**: Do NOT start this step until every model in Step 3 is finished

#### Verification Gate (must pass before Step 5)

| # | Check | How to Verify |
|---|-------|---------------|
| 1 | All 03-x pickle files loaded | Print model count and names |
| 2 | Score summary table complete | All models x (default, tuned) scores present |
| 3 | Best single model identified | Lowest tuned RMSLE highlighted |
| 4 | Prediction correlation computed | Spearman/Pearson matrix plotted |
| 5 | Segment analysis completed | At least DOW and visitor-range segments analyzed |
| 6 | Feature importance compared | Top-20 features compared across models |
| 7 | Ablation study completed | Category-level impact quantified |
| 8 | Ensemble tested | At least 2 strategies (simple avg, weighted) evaluated |
| 9 | Best ensemble RMSLE documented | Final ensemble score printed |
| 10 | Intermediate data saved | `04_comparison_results.pkl` exists |

---

### Step 5: 05_Summary

- **File**: `05_まとめ_{date}.ipynb`
- **Content**:
  - Overall competition retrospective
  - Final model selection rationale
  - Key learnings from each step
  - Future improvement proposals
- **Prerequisite**: Step 4 must be completed
- **IMPORTANT**: This is the LAST notebook to be created. Never create it prematurely.

#### Verification Gate (final)

| # | Check | How to Verify |
|---|-------|---------------|
| 1 | All previous steps referenced | Links/references to 01-04 findings |
| 2 | Final model/ensemble documented | Clear statement of what was submitted |
| 3 | CV vs LB comparison | Gap analysis with sigma interpretation |
| 4 | Lessons learned filled in | `methodology.md` appendix updated |

---

## 【MANDATORY RULE】Intermediate Data Management Across Notebooks

### Purpose

When working across multiple notebooks, recomputing previous steps every time is inefficient. Each notebook saves its output as intermediate files, and subsequent notebooks load them instead of recalculating.

### Directory Structure

```
competitions/{comp_name}/
├── input/                          # Raw data (downloaded from Kaggle)
├── output/                         # Final submission files (submission.csv, etc.)
├── notebooks/
│   └── 説明用資料/
│       ├── intermediate/           # Intermediate data storage
│       │   ├── 01_eda_results.pkl
│       │   ├── 02_feature_design.pkl
│       │   ├── 03-1_lgbm_results.pkl
│       │   └── ...
│       ├── 01_EDA分析_日本語版.ipynb
│       ├── 02_目的変数と説明変数の設計_{date}.ipynb
│       └── ...
```

### Rules

#### 1. Save intermediate data at the end of each notebook

```python
import pickle
INTERMEDIATE_DIR = Path('./intermediate')
INTERMEDIATE_DIR.mkdir(exist_ok=True)

results = {
    'train_features': train_tsr,
    'valid_features': valid_tsr,
    'feature_columns': all_features,
    'grid_df': grid_df,
    'confirmed_settings': confirmed_settings,
    'scores': {
        'score_time': score_time,
        'score_store': score_store,
        'score_all': score_all,
    },
}
with open(INTERMEDIATE_DIR / '02_feature_design.pkl', 'wb') as f:
    pickle.dump(results, f)
```

#### 2. Load intermediate data at the top of subsequent notebooks

```python
import pickle
INTERMEDIATE_DIR = Path('./intermediate')

with open(INTERMEDIATE_DIR / '02_feature_design.pkl', 'rb') as f:
    prev = pickle.load(f)

train_tsr = prev['train_features']
valid_tsr = prev['valid_features']
all_features = prev['feature_columns']
confirmed_settings = prev['confirmed_settings']
```

#### 3. Naming Convention

| Notebook | Intermediate File | Contents |
|----------|------------------|----------|
| 01_EDA | `01_eda_results.pkl` | Key statistics, findings summary |
| 02_Design | `02_feature_design.pkl` | Feature DataFrames, column lists, scores, `confirmed_settings` |
| 03-1_LGBM | `03-1_lgbm_results.pkl` | Model, predictions, scores, importance |
| 03-2_XGB | `03-2_xgb_results.pkl` | Same as above |
| 03-3_RF | `03-3_rf_results.pkl` | Same as above |
| 03-4_CatBoost | `03-4_catboost_results.pkl` | Same as above |
| 04_Comparison | `04_comparison_results.pkl` | All model scores, ensemble results |

#### 4. Large Data Handling

- Use `.parquet` format for DataFrames exceeding 100MB
- Use `.joblib` for model objects (more efficient than pickle)

```python
# Large DataFrames
train_tsr.to_parquet(INTERMEDIATE_DIR / '02_train_features.parquet')

# Model persistence
import joblib
joblib.dump(model, INTERMEDIATE_DIR / '03-1_lgbm_model.joblib')
```

#### 5. When to Recompute

| Situation | Action |
|-----------|--------|
| Intermediate file exists from previous step | Load and use as-is |
| Previous step was modified | Regenerate that step's intermediate file and re-run all downstream notebooks |
| Intermediate file does not exist | Run the previous step's notebook to generate it |

#### 6. gitignore

The `intermediate/` directory contents are excluded from git (pkl / parquet / joblib are all covered by existing gitignore rules).

---

## 【MANDATORY RULE】Verification Protocol

### Before Starting Any Step

1. **Check this guide** for the current step's prerequisites
2. **Verify all upstream intermediate files exist** in `intermediate/`
3. **Load and spot-check upstream data**: print shape, sample rows, key counts

### After Completing Any Step

1. **Run the verification gate checklist** for that step
2. **Verify intermediate file was saved** with correct keys
3. **Update the status table** below
4. **Only then** proceed to the next step

### When Resuming After Interruption

1. Check the status table below to identify the last completed step
2. Verify the intermediate file for that step exists and is valid
3. Resume from the next incomplete step
4. Do NOT skip verification gates even when resuming

---

## 【RULES — MUST FOLLOW】

1. **Never skip steps**: Always proceed in order 01 -> 02 -> 03 -> 04 -> 05
2. **Verify prerequisites**: Confirm each step's verification gates are passed before starting the next
3. **Date suffix**: All filenames must include the date suffix (e.g., `_20260308`)
4. **Japanese content**: All markdown cells and code comments must be written in Japanese
5. **Output location**: Each competition's materials go in `competitions/{comp_name}/notebooks/説明用資料/`
6. **Resuming after interruption**: Check this guide and resume from the incomplete step
7. **Completion check**: Update the status in the competition-specific tracking when each step is completed
8. **confirmed_settings propagation**: All 03-x notebooks MUST load and apply `confirmed_settings` from 02
9. **No hardcoded parameters**: Training period, NaN strategy, and Rolling config must come from `confirmed_settings`, never hardcoded in 03-x or 04

---

## Per-Competition Status

### recruit-restaurant-visitor-forecasting

| Step | File | Status | Verification | 要修正事項 |
|------|------|--------|-------------|-----------|
| 01 | `01_EDA分析_日本語版.ipynb` | ✅ Completed | ✅ Gates passed | — |
| 02 | `02_目的変数と説明変数の設計_20260308.ipynb` | ✅ Completed | ✅ Re-run completed (5フォールドCV) | — |
| 03-1 | `03-1_LightGBMモデル_20260308.ipynb` | ✅ Executed | ❌ 要修正 | キー不一致, CV戦略不整合 |
| 03-2 | `03-2_XGBoostモデル_20260308.ipynb` | ⬜ 要修正 | ❌ 要修正 | キー不一致, CV戦略不整合, NaN未適用 |
| 03-3 | `03-3_RandomForestモデル_20260308.ipynb` | ⬜ 要修正 | ❌ 要修正 | キー不一致, CV戦略不整合 |
| 03-4 | `03-4_CatBoostモデル_20260308.ipynb` | ⬜ 要修正 | ❌ 要修正 | キー不一致, CV戦略不整合, NaN未適用 |
| 04 | `04_各モデルの性能比較_20260308.ipynb` | ⬜ Pending | ⬜ Pending upstream | — |
| 05 | `05_まとめ_20260308.ipynb` | ⬜ Not Started | — | — |

#### 03-x 共通の修正事項（全モデル）

1. **confirmed_settingsキー不一致**: `TRAIN_START` → `best_train_start`, `nan_strategy` → `best_nan_strategy` に修正
2. **CV戦略の統一**: `TimeSeriesSplit(n_splits=3)` → 02の`val_folds`（5フォールド）に変更
3. **NaN戦略の適用**: `best_nan_strategy`を読み込み、モデルごとに適用（RF: 常に-999）
4. **学習データフィルタ**: `best_train_start`で確実にフィルタリング
5. **OptunaをCV平均で最適化**: objective関数でval_foldsベースのCV平均を返す
6. **ルールベースベースライン追加**: store×DOW中央値のRMSLEを算出し比較

#### スコア見通し（外部評価ベース）

| フェーズ | 期待RMSLE |
|---------|----------|
| 時間特徴量のみ（ベースライン） | 0.65〜0.70 |
| +店舗統計・Rolling（現設計） | 0.51〜0.54 |
| +Optuna + GW対応CV | 0.49〜0.52 |
| +アンサンブル（ルールベース混合） | 0.48〜0.50 |

- 金メダル圏（0.502前後）: CVフォールド修正+ルールベースアンサンブルが鍵
- 銀メダル圏（0.51〜0.52）: 現設計を正しく実行すれば到達可能
