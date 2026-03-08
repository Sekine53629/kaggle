# Methodology: EDA and Model Selection Procedure

> A reusable, step-by-step guide for Kaggle tabular/time-series competitions.
> Update this document as new techniques are validated.

| Version | Date       | Author | Notes                          |
|---------|------------|--------|--------------------------------|
| v1.0    | 2026-03-08 | —      | Initial version (Recruit comp) |
| v2.0    | 2026-03-08 | —      | Promoted to root; added verification gates and practical know-how |

---

## Table of Contents

1. [Phase 1: Data Profiling](#phase-1-data-profiling)
2. [Phase 2: Missing Data & Lifecycle Analysis](#phase-2-missing-data--lifecycle-analysis)
3. [Phase 3: Target Variable Analysis](#phase-3-target-variable-analysis)
4. [Phase 4: Temporal Pattern Analysis](#phase-4-temporal-pattern-analysis)
5. [Phase 5: Holiday & Special Period Analysis](#phase-5-holiday--special-period-analysis)
6. [Phase 6: Categorical & Spatial Analysis](#phase-6-categorical--spatial-analysis)
7. [Phase 7: External / Auxiliary Data Analysis](#phase-7-external--auxiliary-data-analysis)
8. [Phase 8: Train/Test Distribution Shift](#phase-8-traintest-distribution-shift)
9. [Phase 9: Validation Strategy Design](#phase-9-validation-strategy-design)
10. [Phase 10: Feature Engineering Candidates](#phase-10-feature-engineering-candidates)
11. [Phase 11: Model Selection Procedure](#phase-11-model-selection-procedure)
12. [Verification Know-How](#verification-know-how)
13. [Appendix: Lessons Learned](#appendix-lessons-learned)

---

## Phase 1: Data Profiling

### Objectives
- Understand shape, dtypes, and memory usage of every table.
- Identify join keys and relational structure across tables.
- Detect obvious data quality issues (duplicates, constant columns).

### Checklist
- [ ] Print shape, dtypes, and head(5) for each table.
- [ ] Check for duplicate rows and duplicate primary keys.
- [ ] Verify join key coverage (e.g., how many IDs in table A exist in table B).
- [ ] Summarize memory usage; apply `reduce_mem_usage()` if needed.

### Key Questions
- Which tables can be joined and on what keys?
- Are there any tables with surprisingly few or many rows?
- Are date columns parsed correctly?

---

## Phase 2: Missing Data & Lifecycle Analysis

### Objectives
- Distinguish between **structural absence** (store was closed / not yet open) and **random missing** data.
- Identify store lifecycle events: opening date, closing date, temporary closures.

### Checklist
- [ ] For each entity (e.g., store), compute the first and last observation date.
- [ ] Build a calendar grid (entity x date) and flag gaps.
- [ ] Classify gaps:
  - **Pre-opening**: dates before the entity's first observation.
  - **Post-closing**: dates after the entity's last observation (no further data).
  - **Mid-life gap**: dates between first and last observation with no record.
- [ ] Analyze mid-life gaps:
  - Duration distribution (1 day vs. weeks).
  - Day-of-week pattern (e.g., always closed on Tuesdays -> regular holiday).
  - Correlation with public holidays or special periods.
- [ ] Visualize: heatmap of entity x week showing data presence.

### Pitfalls
- **Do not impute zeros for structural gaps.** A store that is closed has no visitors — this is different from zero visitors.
- Regular weekly closures (e.g., every Monday) should be separated from irregular gaps.

---

## Phase 3: Target Variable Analysis

### Objectives
- Understand the distribution, skewness, and outlier structure of the target.
- Determine the appropriate transformation (log, Box-Cox, etc.).

### Checklist
- [ ] Basic statistics: mean, median, std, skewness, kurtosis.
- [ ] Histogram of raw values and log-transformed values.
- [ ] Outlier detection: IQR method, z-score, or domain-specific thresholds.
- [ ] Per-entity target statistics: mean, std, coefficient of variation (CV).
- [ ] Verify alignment with evaluation metric:
  - RMSLE -> log1p transform is natural.
  - RMSE -> consider whether raw or transformed target is better.

### Key Questions
- Is the distribution heavy-tailed? If so, log-transform.
- Are outliers genuine (special events) or data errors?
- Does the variance scale with the mean (heteroscedasticity)?

---

## Phase 4: Temporal Pattern Analysis

### Objectives
- Decompose time series into trend, seasonality, and residuals.
- Quantify autocorrelation to determine useful lag features.

### Checklist
- [ ] Daily aggregated time series plot (sum, mean, count).
- [ ] Day-of-week effect: bar chart of mean target by DOW.
- [ ] Month effect: bar chart of mean target by month.
- [ ] DOW x Month heatmap.
- [ ] STL decomposition (or seasonal_decompose) on aggregated series.
- [ ] ACF / PACF plots to identify significant lags.
- [ ] Year-over-year comparison (if multiple years available).

### Key Outputs
- Which lags are significant? (e.g., lag-7 for weekly seasonality)
- Is there a trend component? (growth or decline over time)
- Seasonal amplitude: is it additive or multiplicative?

---

## Phase 5: Holiday & Special Period Analysis

### Objectives
- Go beyond binary holiday flags — analyze **raw behavioral data** during special periods.
- Understand per-entity variation in holiday response.

### Step 5a: Define Special Periods from Domain Knowledge

For Japan-based competitions, key periods include:

| Period          | Approximate Dates     | Expected Behavior          |
|-----------------|-----------------------|----------------------------|
| New Year        | Dec 31 - Jan 3        | Many closures; open stores may see surge |
| Golden Week     | Apr 29 - May 7        | Major holiday; varies by store type |
| Obon            | Aug 13 - Aug 16       | Regional holiday; travel season |
| Silver Week     | Sep (varies by year)  | Shorter holiday cluster     |
| Year-end parties| Dec 1 - Dec 28        | Izakaya/bar surge           |
| Christmas       | Dec 23 - Dec 25       | Couples/families dining out |

**Important**: Define these from raw calendar data, not just `holiday_flg`.

### Step 5b: Per-Store Holiday Behavior
- [ ] For each special period, compute per-store metrics:
  - Did the store operate? (presence of data)
  - If operated, what was the visitor ratio vs. the store's normal level?
- [ ] Cluster stores by holiday behavior pattern (e.g., "closes for New Year", "surges during GW").

### Step 5c: Adjacent-Day Effects
- [ ] Compute "day before holiday" and "day after holiday" effects.
- [ ] These are often more significant than the holiday itself (e.g., Friday night before a long weekend).

### Step 5d: Rolling Holiday Effect
- [ ] Compare the same holiday across different years or occurrences.
- [ ] Is the holiday effect stable, growing, or declining?

### Pitfalls
- **Do not treat all holidays equally.** New Year and a random Monday holiday have completely different effects.
- **Chain stores may differ.** Even stores in the same chain/genre react differently based on location.

---

## Phase 6: Categorical & Spatial Analysis

### Objectives
- Understand how categorical features (genre, area) relate to the target.
- Identify spatial patterns and clusters.

### Checklist
- [ ] Category-level statistics: count, mean target, std, CV.
- [ ] Genre x DOW interaction: do some genres have stronger weekend effects?
- [ ] Area analysis: prefecture / city level aggregation.
- [ ] Geographic scatter plot (lat/lon colored by mean target).
- [ ] Cross-entity correlation: do stores in the same area/genre correlate?

---

## Phase 7: External / Auxiliary Data Analysis

### Objectives
- Evaluate the predictive value of auxiliary data (reservations, external datasets).
- Determine coverage and reliability.

### Checklist
- [ ] Coverage analysis: what fraction of entities/dates have auxiliary data?
- [ ] Correlation between auxiliary signals and target.
- [ ] Lead time analysis (for reservation-type data):
  - How far in advance are reservations made?
  - Does lead time correlate with party size or target?
- [ ] If multiple auxiliary sources exist, check overlap and consistency.

### Pitfalls
- Low-coverage auxiliary data can still be useful but requires careful imputation.
- Reservation data may have **look-ahead bias** if not properly time-bounded.

---

## Phase 8: Train/Test Distribution Shift

### Objectives
- Verify that the test set is not fundamentally different from training data.
- Identify potential domain shift risks.

### Checklist
- [ ] Entity overlap: are all test entities present in training?
- [ ] Temporal gap: is there a gap between train and test periods?
- [ ] Feature distribution comparison (KS test or visual):
  - DOW distribution in train vs. test.
  - Holiday distribution in train vs. test.
  - Genre/area distribution in train vs. test.
- [ ] If test period includes special events not in train -> flag as risk.

### Key Question for This Step
- **Does the test period contain a special period (e.g., Golden Week) that the model has only seen once or never?** If so, special handling is required.

---

## Phase 9: Validation Strategy Design

### Objectives
- Design a CV strategy that mimics the train/test split as closely as possible.
- Prevent temporal leakage.

### Principles
1. **Never use random split for time series.** Use `TimeSeriesSplit` or custom temporal folds.
2. **Match the test window length.** If the test period is 39 days, each validation fold should also be ~39 days.
3. **Include at least one fold with a special period** (e.g., GW or year-end) if the test period contains one.
4. **Gap between train and validation** may be needed if features use recent lags.

### Recommended Approach: Sliding Window CV

```
Fold 1: Train [.......] -> Val [---39 days---]
Fold 2: Train [...........] -> Val [---39 days---]
Fold 3: Train [..............] -> Val [---39 days---]
                                     ^ include GW-like period if possible
```

### Checklist
- [ ] Define fold boundaries (start/end dates for train and validation in each fold).
- [ ] Verify no feature leakage across the boundary.
- [ ] Compare CV score vs. LB score — if they diverge, the CV strategy needs adjustment.
- [ ] Document the exact fold dates for reproducibility.

### Overfitting Signals
- CV score improving but LB score stagnating or worsening.
- Large variance across CV folds.
- Model relying heavily on entity-specific memorization.

---

## Phase 10: Feature Engineering Candidates

### Objectives
- Consolidate all EDA findings into actionable feature ideas.

### Template

| # | Feature Name         | Source         | Type        | Rationale                         | Priority |
|---|----------------------|---------------|-------------|-----------------------------------|----------|
| 1 | `dow`                | visit_date     | Categorical | Strong weekly pattern              | High     |
| 2 | `month`              | visit_date     | Categorical | Monthly seasonality                | High     |
| 3 | `is_holiday`         | date_info      | Binary      | Holiday effect                     | High     |
| 4 | `is_before_holiday`  | date_info      | Binary      | Pre-holiday surge                  | High     |
| 5 | `store_mean_visitors`| air_visit      | Numeric     | Store-level baseline               | High     |
| 6 | `genre`              | air_store      | Categorical | Genre affects scale                | Medium   |
| 7 | `lag_7`, `lag_14`    | air_visit      | Numeric     | Weekly autocorrelation             | High     |
| 8 | `rolling_mean_7d`    | air_visit      | Numeric     | Recent trend                       | High     |
| 9 | `reserve_visitors`   | air/hpg_reserve| Numeric     | Leading indicator                  | Medium   |
| 10| `special_period_flag`| domain knowledge| Categorical| GW/NY/Obon behavior               | High     |

---

## Phase 11: Model Selection Procedure

### Step 1: Baseline（ルールベース）
- **店舗×曜日の過去中央値**が最も重要なベースライン。上位解法で繰り返し確認されている知見：「store×DOW中央値がGBDTに匹敵し、アンサンブルで補完効果がある」
- このベースラインのRMSLEを出してから、ML モデルの改善幅を測ることでスコアの位置づけが明確になる

```python
# 最強のルールベースベースライン
store_dow_median = train_df.groupby(
    ['air_store_id', 'dow'])['visitors'].median()
baseline_pred = valid_df.apply(
    lambda r: store_dow_median.get((r['air_store_id'], r['dow']),
                                    train_df['visitors'].median()), axis=1)
baseline_rmsle = rmsle(valid_df['visitors'], baseline_pred)
```

### Step 2: Single-Model Comparison
- Train each candidate model with the same feature set and CV strategy.
- Candidates (tabular):
  - LightGBM (fast, handles categoricals natively)
  - XGBoost (slightly different regularization)
  - CatBoost (good with high-cardinality categoricals)
  - Random Forest (robust but slower)
- Candidates (time-series specific):
  - Prophet / NeuralProphet
  - ARIMA / SARIMAX (per-entity)

### Step 3: Hyperparameter Tuning
- Use Optuna or similar for Bayesian optimization.
- Tune on CV score, not on a single fold.
- Key parameters to tune:
  - Learning rate + n_estimators (inverse relationship)
  - max_depth / num_leaves
  - Regularization (lambda, alpha, min_child_samples)
  - Subsampling (feature and row)

### Step 4: Ensemble
- Blend top 2-3 models (weighted average or stacking).
- Verify that ensemble improves CV score.

### Step 5: Final Validation & Submission Pipeline
- Retrain on full training data with best hyperparameters.
- Sanity-check predictions: distribution, range, special period behavior.
- **テストデータへの予測パイプラインを実装**: `sample_submission.csv`のID形式（`air_store_id_YYYY-MM-DD`）に合わせた予測生成

```python
# テストデータへの予測パイプラインの例
submission = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
submission['air_store_id'] = submission['id'].str[:-11]
submission['visit_date'] = pd.to_datetime(submission['id'].str[-10:])

# テスト期間の特徴量を構築（02と同一パイプライン）
# → 予測 → submission.csv に保存
```

### Step 6: Ensemble with Rule-Based Prediction
- **ルールベース予測（store×DOW中央値）とMLモデルのアンサンブル**
- 単純平均でもアンサンブル効果があるが、重み付き平均（Optuna or Nelder-Mead）が最適
- 残差相関が低いモデル同士の組み合わせが最も効果的

```python
# ルールベース + LightGBM のアンサンブル例
final_pred = w_rule * baseline_pred + w_lgb * lgb_pred
# w_rule, w_lgb を Optuna or scipy.optimize で最適化
```

---

## Verification Know-How

This section documents reusable verification patterns applicable across competitions. Each pattern includes a code snippet and interpretation guide.

### V1: Feature Leakage Detection

```python
# Check if any feature has impossibly high correlation with target
corr = train_df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
suspicious = corr[corr.abs() > 0.95]
if len(suspicious) > 0:
    print("WARNING: Possible leakage features:")
    print(suspicious)
```

**Interpretation**: Correlation > 0.95 with target likely indicates leakage. Investigate the feature's construction — does it use future information?

### V2: Temporal Leakage in Rolling Features

```python
# Verify rolling features don't peek into the future
for col in rolling_cols:
    # For each date, check that rolling value only uses past data
    sample = df.groupby('store_id').apply(
        lambda g: g.sort_values('visit_date').head(10)
    )
    # Rolling features on the first few dates should be NaN or use minimal data
    early_nans = sample[col].isna().sum()
    print(f"{col}: {early_nans} NaN in early dates (expected > 0)")
```

### V3: CV-LB Alignment Check

```python
# After submission, compare CV and LB scores
cv_scores = [fold1_score, fold2_score, fold3_score]
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)
lb_score = 0.xxx  # from Kaggle

gap = abs(cv_mean - lb_score)
ratio = gap / cv_std if cv_std > 0 else float('inf')
print(f"CV: {cv_mean:.5f} +/- {cv_std:.5f}")
print(f"LB: {lb_score:.5f}")
print(f"Gap: {gap:.5f} ({ratio:.1f} sigma)")
# If gap > 2 sigma, CV strategy needs revision
```

### V4: Prediction Sanity Check

```python
# Verify predictions are in reasonable range
pred_stats = pd.Series(predictions)
train_stats = train_df[target_col]

checks = {
    'pred_min >= 0': pred_stats.min() >= 0,
    'pred_max < 10x train_max': pred_stats.max() < 10 * train_stats.max(),
    'pred_mean within 2x train_mean': abs(pred_stats.mean() - train_stats.mean()) < 2 * train_stats.mean(),
    'no NaN in predictions': pred_stats.isna().sum() == 0,
}
for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check}")
```

### V5: NaN Strategy Validation

```python
# Compare multiple NaN handling strategies
strategies = {
    'fill_zero': df[col].fillna(0),
    'fill_median': df[col].fillna(df[col].median()),
    'forward_fill': df.groupby('store_id')[col].ffill(),
    'leave_nan': df[col],  # let model handle
}
for name, series in strategies.items():
    score = evaluate_with_strategy(series)
    print(f"{name}: RMSLE = {score:.5f}")
```

### V6: Rolling Window Configuration Test

```python
# Test different rolling configurations
configs = {
    'calendar_7d': lambda g: g.rolling(7, min_periods=1).mean(),
    'calendar_14d': lambda g: g.rolling(14, min_periods=1).mean(),
    'business_7d': lambda g: g[g['is_open']].rolling(7, min_periods=1).mean(),
    'ewm_7d': lambda g: g.ewm(span=7, min_periods=1).mean(),
}
# Evaluate each and pick the best
```

### V7: Ensemble Weight Stability

```python
# Check if ensemble weights are stable across CV folds
fold_weights = []
for fold in range(n_folds):
    weights = optimize_weights(fold_predictions[fold])
    fold_weights.append(weights)

weight_df = pd.DataFrame(fold_weights, columns=model_names)
print("Weight stability across folds:")
print(weight_df.describe())
# High std in weights -> unstable ensemble, may overfit
```

### V8: Business Day vs Calendar Day Rolling Comparison

```python
# Compare business-day-based and calendar-day-based rolling statistics
# to verify the impact of closure handling
for store_id in sample_stores:
    store_data = df[df['store_id'] == store_id].sort_values('visit_date')
    calendar_roll = store_data['visitors'].rolling(7, min_periods=1).mean()
    business_roll = store_data.loc[store_data['is_open'], 'visitors'].rolling(7, min_periods=1).mean()
    diff = (calendar_roll - business_roll).abs().mean()
    print(f"Store {store_id}: avg diff = {diff:.2f}")
```

---

## Phase 12: Model Training Pipeline (03-x Notebooks)

### Objectives
- 各モデルノートブック（03-x）で、02の`confirmed_settings`を正確に反映する
- CV戦略を02と統一し、スコアの比較可能性を担保する
- 中間データの受け渡しでキー名の不整合を防ぐ

### 12.1: confirmed_settings の正確な読み込み

```python
# 02_feature_design.pkl からの読み込みパターン
with open(INTERMEDIATE_DIR / '02_feature_design.pkl', 'rb') as f:
    prev_02 = pickle.load(f)

confirmed = prev_02['confirmed_settings']

# ★ キー名は02の保存時と完全一致させること
best_train_start = confirmed['best_train_start']         # 絶対日付（例: '2015-12-12'）
best_train_period_days = confirmed['best_train_period_days']  # 相対日数（例: 456）
best_nan_strategy = confirmed['best_nan_strategy']       # 例: 'NaN(デフォルト)'
best_rolling_config = confirmed['best_rolling_config']   # 例: '現行のみ'

# 学習データのフィルタリング
train_df = prev_02['train_features']
train_df = train_df[train_df['visit_date'] >= best_train_start].reset_index(drop=True)
```

**よくある間違い:**
- `confirmed_settings.get('TRAIN_START')` → 02では `best_train_start` で保存
- `confirmed_settings.get('nan_strategy')` → 02では `best_nan_strategy` で保存
- キー不一致は `None` を返しサイレントに失敗する → **必ず KeyError で落とすか、ロード直後に全キーをprint確認する**

### 12.2: NaN戦略の適用

```python
# NaN戦略の適用
if best_nan_strategy == 'NaN(デフォルト)':
    # LightGBM / XGBoost / CatBoost: NaN対応 → そのまま
    X_train = train_df[all_features]
    X_valid = valid_df[all_features]
elif best_nan_strategy == '-999埋め':
    X_train = train_df[all_features].fillna(-999)
    X_valid = valid_df[all_features].fillna(-999)
elif best_nan_strategy == '0埋め':
    X_train = train_df[all_features].fillna(0)
    X_valid = valid_df[all_features].fillna(0)
elif best_nan_strategy == '中央値埋め':
    medians = train_df[all_features].median()
    X_train = train_df[all_features].fillna(medians)
    X_valid = valid_df[all_features].fillna(medians)

# ★ RandomForestはNaN非対応 → best_nan_strategyが'NaN(デフォルト)'でも-999埋めが必要
if model_type == 'RandomForest':
    X_train = X_train.fillna(-999)
    X_valid = X_valid.fillna(-999)
```

### 12.3: CV戦略の統一

**全ノートブックで同一のCV戦略を使用する。** 02で設計した`val_folds`（01 EDAベース）を03-xでも使う。

```python
# 02から val_folds を読み込み
val_folds = prev_02['val_folds']

# CV評価
cv_scores = []
for fold in val_folds:
    val_start = pd.Timestamp(fold['val_start'])
    val_end = pd.Timestamp(fold['val_end'])
    train_mask = (full_df['visit_date'] < val_start)
    valid_mask = (full_df['visit_date'] >= val_start) & (full_df['visit_date'] <= val_end)
    # ... 学習・評価 ...
```

**使ってはいけないパターン:**
```python
# NG: sklearn の TimeSeriesSplit は val_folds と異なるフォールド境界になる
tscv = TimeSeriesSplit(n_splits=3)  # ← 02の5フォールドCVと比較不能
```

### 12.4: Optuna チューニングの注意点

- **Single Split でチューニングすると、その分割に過適合する** — 50 trialsで特定の検証期間に偏ったパラメータが選ばれるリスク
- **CV平均でチューニングを推奨**（計算コストが高い場合はtrial数を減らして対応）
- `iterations`（ブーストラウンド数）はベースラインと揃える（2000）。Optunaでは3000にしない

```python
# ★推奨: objective関数内でCVスコアを返す
def objective(trial):
    params = { ... }
    cv_scores = []
    for fold in val_folds:
        train_mask = df['visit_date'] < fold['val_start']
        val_mask = (df['visit_date'] >= fold['val_start']) & (df['visit_date'] <= fold['val_end'])
        # フォールドごとにfit & evaluate
        ...
        cv_scores.append(fold_score)
    return np.mean(cv_scores)
```

### 12.5: 中間データの保存形式

```python
# 03-x で保存すべきキー（統一形式）
results = {
    # 予測値（single split）
    'valid_pred': pred,
    'valid_pred_log': pred_log,
    'valid_actual': valid_df['visitors'].values,
    'residuals': residuals,
    # スコア
    'score_single': score_single,
    'cv_scores': cv_scores,            # val_foldsベースのCV
    'cv_mean': np.mean(cv_scores),
    'cv_std': np.std(cv_scores),
    # モデル情報
    'params': default_params,
    'best_iteration': best_iter,
    'feature_importance': importance_df,
    # Optunaチューニング結果
    'tuned_params': tuned_params,
    'tuned_score': tuned_score,
    'tuned_pred': tuned_pred,
    'tuned_pred_log': tuned_pred_log,
    'optuna_best_params': study.best_params,
}
```

### Checklist
- [ ] `confirmed_settings`のキー名が02の保存形式と完全一致
- [ ] `best_train_start`で学習データがフィルタリングされている
- [ ] `best_nan_strategy`が適用されている（RF例外あり）
- [ ] CV戦略が02の`val_folds`と同一
- [ ] Optunaのiterationsがベースラインと揃っている
- [ ] 保存する中間データのキーが統一形式に従っている

---

## Phase 13: CV-LB Alignment Analysis

### Objectives
- CV平均とLBスコアの乖離を定量化し、CV戦略の妥当性を評価する

### 分析手順

```python
# 提出後のCV-LB比較
cv_mean = 0.545
cv_std = 0.026
public_lb = 0.480
private_lb = 0.575

# 乖離分析
public_gap = abs(cv_mean - public_lb)
private_gap = abs(cv_mean - private_lb)
public_sigma = public_gap / cv_std
private_sigma = private_gap / cv_std

print(f'CV: {cv_mean:.5f} ± {cv_std:.5f}')
print(f'Public LB: {public_lb:.5f} (乖離: {public_gap:.5f} = {public_sigma:.1f}σ)')
print(f'Private LB: {private_lb:.5f} (乖離: {private_gap:.5f} = {private_sigma:.1f}σ)')
```

### 乖離の解釈

| 乖離 | 意味 | 対処 |
|------|------|------|
| < 1σ | CV戦略は妥当 | そのまま継続 |
| 1-2σ | 軽度の不整合 | テスト期間の特殊性を確認 |
| > 2σ | CV戦略の見直しが必要 | フォールド設計・特徴量リーク確認 |
| Public < CV < Private | Public過適合の可能性 | Privateに近いフォールドを重視 |

### Recruit Competition の実例
- CV平均: 0.545, Public LB: 0.480, Private LB: 0.575
- Public乖離: 2.5σ → テスト期間（GW含む特殊期間）がCVフォールドと異なる分布
- Private乖離: 1.2σ → やや乖離するが許容範囲
- **教訓**: テスト期間にGWが含まれる場合、GWを含むフォールドの重みを上げるべき

---

## Appendix: Lessons Learned

> Add entries here after each competition. This section grows over time.

### Recruit Restaurant Visitor Forecasting (2026-03-08)

- **Lesson 1: confirmed_settings のキー名整合性が重要**
  - 02で `best_train_start` で保存したのに、03-xで `TRAIN_START` で読み込むとサイレントに `None` が返る
  - 対策: ロード直後に全キーをprint、またはキー存在チェックで `KeyError` を発生させる

- **Lesson 2: CV戦略は全ノートブックで統一する**
  - 02で5フォールド時系列CVを設計したのに、03-xで `TimeSeriesSplit(n_splits=3)` を使うとスコアが比較不能
  - 対策: 02の `val_folds` を03-xでも読み込んで使用する

- **Lesson 3: 店舗データの欠損は行単位で発生する**
  - `air_visit_data.csv` は来客日のみ記録。欠損日は行ごと存在しない（NaNではない）
  - フォールドごとに店舗の観測数が異なるため、最低観測数フィルタ（MIN_STORE_RECORDS=7）が必要

- **Lesson 4: 学習期間の検証はリーク防止が必須**
  - 絶対日付（例: `2016-07-01~`）ではなく、`val_start`からの相対日数で管理する
  - 「全期間」はフォールドごとに学習量が異なり公平比較できない

- **What worked**: ジャンル交互作用特徴量、営業日ベースRolling、有意差判定（±1σ）による設定選択
- **What didn't work**: 単一train/valid分割での設定決定（過適合リスク）
- **Key takeaway**: CV戦略は最初（01 EDA）に設計し、全ノートブックで一貫して使用する

---

*This document is a living reference. Update it after each competition with new insights.*
