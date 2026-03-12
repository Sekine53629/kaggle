"""Diagnose prediction downward bias"""
import sys, io, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, pandas as pd, pickle, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_squared_error

SEED = 42
np.random.seed(SEED)
INPUT_DIR = Path('input')
INTERMEDIATE_DIR = Path('notebooks/説明用資料/intermediate')

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

with open(INTERMEDIATE_DIR / '02_feature_design.pkl', 'rb') as f:
    prev = pickle.load(f)

train_df = prev['train_features']
valid_df = prev['valid_features']
all_features = prev['feature_columns']['all_features']
settings = prev['confirmed_settings']
val_folds = prev['val_folds']
TRAIN_START = settings['best_train_start']
train_df = train_df[train_df['visit_date'] >= TRAIN_START].copy()
full_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values('visit_date').reset_index(drop=True)

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.02, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_samples': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

# === 1. Validation bias by day-of-week ===
print('=== 1. Validation bias by DOW ===')
dtrain = lgb.Dataset(train_df[all_features], label=np.log1p(train_df['visitors']))
dvalid = lgb.Dataset(valid_df[all_features], label=np.log1p(valid_df['visitors']), reference=dtrain)
model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
valid_pred = np.expm1(model.predict(valid_df[all_features]))

valid_df = valid_df.copy()
valid_df['pred'] = np.clip(valid_pred, 1, None)
valid_df['bias'] = valid_df['pred'] - valid_df['visitors']
valid_df['bias_pct'] = (valid_df['pred'] - valid_df['visitors']) / valid_df['visitors'] * 100

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for dow in range(7):
    mask = valid_df['dow'] == dow
    if not mask.any(): continue
    actual = valid_df.loc[mask, 'visitors'].mean()
    pred_m = valid_df.loc[mask, 'pred'].mean()
    bias_pct = (pred_m - actual) / actual * 100
    n = mask.sum()
    print(f'  {dow_names[dow]}: actual={actual:.1f}, pred={pred_m:.1f}, bias={bias_pct:+.1f}%, n={n}')

# === 2. Validation bias by date ===
print('\n=== 2. Validation bias by date ===')
by_date = valid_df.groupby('visit_date').agg(
    actual_mean=('visitors', 'mean'),
    pred_mean=('pred', 'mean'),
    n=('visitors', 'count')
).reset_index()
by_date['bias_pct'] = (by_date['pred_mean'] - by_date['actual_mean']) / by_date['actual_mean'] * 100
for _, r in by_date.iterrows():
    d = r['visit_date'].strftime('%Y-%m-%d')
    dow = r['visit_date'].dayofweek
    print(f'  {d} ({dow_names[dow]}): actual={r["actual_mean"]:.1f}, pred={r["pred_mean"]:.1f}, bias={r["bias_pct"]:+.1f}%')

# === 3. Frozen features analysis ===
print('\n=== 3. Frozen features impact ===')
grid_df = prev['grid_df']
rlc = [c for c in grid_df.columns if any(c.startswith(p) for p in [
    'rolling_mean_', 'rolling_std_', 'ewm_mean', 'lag_', 'open_ratio_', 'closed_streak', 'days_since_long_closure'])]
print(f'  Rolling/lag feature count: {len(rlc)}')

# Check how feature importance distributes between frozen and non-frozen
imp_gain = model.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'feature': all_features, 'gain': imp_gain}).sort_values('gain', ascending=False)
imp_df['pct'] = imp_df['gain'] / imp_df['gain'].sum() * 100

frozen_features = [f for f in all_features if f in rlc]
static_features = [f for f in all_features if f not in rlc]
frozen_imp = imp_df[imp_df['feature'].isin(frozen_features)]['pct'].sum()
static_imp = imp_df[imp_df['feature'].isin(static_features)]['pct'].sum()
print(f'  Frozen features importance: {frozen_imp:.1f}% ({len(frozen_features)} features)')
print(f'  Static features importance: {static_imp:.1f}% ({len(static_features)} features)')
print(f'  Top frozen features:')
for _, r in imp_df[imp_df['feature'].isin(frozen_features)].head(10).iterrows():
    print(f'    {r["feature"]:40s} {r["pct"]:5.1f}%')

# === 4. Test frozen feature values vs validation ===
print('\n=== 4. Frozen features: last train date vs validation mean ===')
ld = grid_df['visit_date'].max()
print(f'  Last date in grid_df: {ld}')
frz = grid_df[grid_df['visit_date'] == ld][['air_store_id'] + rlc].set_index('air_store_id')

# Compare frozen values to mean over validation period for same stores
for feat in frozen_features[:10]:
    if feat in valid_df.columns and feat in frz.columns:
        frozen_mean = frz[feat].mean()
        valid_mean = valid_df[feat].mean()
        if valid_mean != 0:
            diff = (frozen_mean - valid_mean) / abs(valid_mean) * 100
            print(f'  {feat:40s}: frozen={frozen_mean:.2f}, valid_mean={valid_mean:.2f}, diff={diff:+.1f}%')

# === 5. Store-level bias distribution ===
print('\n=== 5. Store-level bias distribution ===')
store_bias = valid_df.groupby('air_store_id').agg(
    actual_mean=('visitors', 'mean'),
    pred_mean=('pred', 'mean'),
    n=('visitors', 'count')
)
store_bias['bias_pct'] = (store_bias['pred_mean'] - store_bias['actual_mean']) / store_bias['actual_mean'] * 100
print(f'  Stores with >20% negative bias: {(store_bias["bias_pct"] < -20).sum()} / {len(store_bias)}')
print(f'  Stores with >20% positive bias: {(store_bias["bias_pct"] > 20).sum()} / {len(store_bias)}')
print(f'  Median store bias: {store_bias["bias_pct"].median():+.1f}%')
print(f'  Mean store bias:   {store_bias["bias_pct"].mean():+.1f}%')

# Percentiles
for p in [10, 25, 50, 75, 90]:
    print(f'  P{p} bias: {store_bias["bias_pct"].quantile(p/100):+.1f}%')

# === 6. Test set analysis ===
print('\n=== 6. Test set structure ===')
sample_sub = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
sample_sub['air_store_id'] = sample_sub['id'].str[:-11]
sample_sub['visit_date'] = pd.to_datetime(sample_sub['id'].str[-10:])
print(f'  Test date range: {sample_sub["visit_date"].min()} to {sample_sub["visit_date"].max()}')
print(f'  Test stores: {sample_sub["air_store_id"].nunique()}')
print(f'  Test rows: {len(sample_sub)}')

# Check DOW distribution in test
sample_sub['dow'] = sample_sub['visit_date'].dt.dayofweek
test_dow = sample_sub.groupby('dow').size()
for dow in range(7):
    print(f'  {dow_names[dow]}: {test_dow.get(dow, 0)} rows')

# GW dates in test
gw_dates = sample_sub[sample_sub['visit_date'].between('2017-04-29', '2017-05-07')]
print(f'  GW period rows: {len(gw_dates)}')

# === 7. Potential fix: iterative prediction ===
print('\n=== 7. Iterative prediction simulation (validation period) ===')
# Instead of frozen features, update lag/rolling day by day
# Compare RMSLE: frozen vs iterative

# For simplicity, test just lag_1 and rolling_mean_7 updates
# Full model with frozen features (current approach)
score_frozen = rmsle(valid_df['visitors'], valid_df['pred'])
print(f'  Frozen approach RMSLE: {score_frozen:.5f}')

# Train on all data except last 39 days, then iteratively predict
# Using the actual validation period
val_dates = sorted(valid_df['visit_date'].unique())
print(f'  Validation dates: {len(val_dates)}')

# Iterative: for each date, update lag features from previous day's prediction
iter_preds = []
prev_day_preds = {}  # store_id -> list of recent predictions

for date in val_dates:
    day_df = valid_df[valid_df['visit_date'] == date].copy()

    # Update lag_1 with previous day's prediction if available
    for idx, row in day_df.iterrows():
        sid = row['air_store_id']
        if sid in prev_day_preds and len(prev_day_preds[sid]) > 0:
            day_df.loc[idx, 'lag_1'] = np.log1p(prev_day_preds[sid][-1])
        if sid in prev_day_preds and len(prev_day_preds[sid]) >= 7:
            recent = prev_day_preds[sid][-7:]
            day_df.loc[idx, 'rolling_mean_7'] = np.mean(np.log1p(recent))

    preds = np.expm1(model.predict(day_df[all_features]))
    preds = np.clip(preds, 1, None)
    iter_preds.extend(preds)

    for idx, (_, row) in enumerate(day_df.iterrows()):
        sid = row['air_store_id']
        if sid not in prev_day_preds:
            prev_day_preds[sid] = []
        prev_day_preds[sid].append(preds[idx])

iter_preds = np.array(iter_preds)
# Note: valid_df order matches date order since we iterate dates
valid_sorted = valid_df.sort_values('visit_date')
score_iterative = rmsle(valid_sorted['visitors'], iter_preds)
print(f'  Iterative approach RMSLE: {score_iterative:.5f}')
print(f'  Difference: {score_frozen - score_iterative:+.5f}')
print(f'  Iterative pred mean: {iter_preds.mean():.1f} vs Frozen pred mean: {valid_df["pred"].mean():.1f}')

print('\nDone!')
