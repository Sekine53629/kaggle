# Methodology: EDA and Model Selection Procedure

> A reusable, step-by-step guide for Kaggle tabular/time-series competitions.
> Update this document as new techniques are validated.

| Version | Date       | Author | Notes                          |
|---------|------------|--------|--------------------------------|
| v1.0    | 2026-03-08 | —      | Initial version (Recruit comp) |

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
12. [Appendix: Lessons Learned](#appendix-lessons-learned)

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
- [ ] Build a calendar grid (entity × date) and flag gaps.
- [ ] Classify gaps:
  - **Pre-opening**: dates before the entity's first observation.
  - **Post-closing**: dates after the entity's last observation (no further data).
  - **Mid-life gap**: dates between first and last observation with no record.
- [ ] Analyze mid-life gaps:
  - Duration distribution (1 day vs. weeks).
  - Day-of-week pattern (e.g., always closed on Tuesdays → regular holiday).
  - Correlation with public holidays or special periods.
- [ ] Visualize: heatmap of entity × week showing data presence.

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
  - RMSLE → log1p transform is natural.
  - RMSE → consider whether raw or transformed target is better.

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
- [ ] DOW × Month heatmap.
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
| New Year        | Dec 31 – Jan 3        | Many closures; open stores may see surge |
| Golden Week     | Apr 29 – May 7        | Major holiday; varies by store type |
| Obon            | Aug 13 – Aug 16       | Regional holiday; travel season |
| Silver Week     | Sep (varies by year)  | Shorter holiday cluster     |
| Year-end parties| Dec 1 – Dec 28        | Izakaya/bar surge           |
| Christmas       | Dec 23 – Dec 25       | Couples/families dining out |

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
- [ ] Genre × DOW interaction: do some genres have stronger weekend effects?
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
- [ ] If test period includes special events not in train → flag as risk.

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
Fold 1: Train [.......] → Val [---39 days---]
Fold 2: Train [...........] → Val [---39 days---]
Fold 3: Train [..............] → Val [---39 days---]
                                     ↑ include GW-like period if possible
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

### Step 1: Baseline
- Start with the simplest reasonable model (e.g., per-store median, or global mean by DOW).
- This sets the floor for CV score.

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

### Step 5: Final Validation
- Retrain on full training data with best hyperparameters.
- Sanity-check predictions: distribution, range, special period behavior.

---

## Appendix: Lessons Learned

> Add entries here after each competition. This section grows over time.

### Recruit Restaurant Visitor Forecasting (2026-03-08)

- **Lesson**: (To be filled after competition)
- **What worked**:
- **What didn't work**:
- **Key takeaway**:

---

*This document is a living reference. Update it after each competition with new insights.*
