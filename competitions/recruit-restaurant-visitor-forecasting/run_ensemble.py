"""LightGBM + XGBoost ensemble pipeline"""
import sys, io, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)
import numpy as np, pandas as pd, pickle, lightgbm as lgb, xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_squared_error

SEED = 42
np.random.seed(SEED)
INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
INTERMEDIATE_DIR = Path('notebooks/説明用資料/intermediate')

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 1, None)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

# === Load data ===
with open(INTERMEDIATE_DIR / '02_feature_design.pkl', 'rb') as f:
    prev = pickle.load(f)

train_df = prev['train_features']
valid_df = prev['valid_features']
all_features = prev['feature_columns']['all_features']
settings = prev['confirmed_settings']
val_folds = prev['val_folds']
TRAIN_START = settings['best_train_start']
train_df = train_df[train_df['visit_date'] >= TRAIN_START].copy()

# === Reserve features ===
print('=== 1. Reserve features ===')
hpg_res = pd.read_csv(INPUT_DIR / 'hpg_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime'])
air_res = pd.read_csv(INPUT_DIR / 'air_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime'])
hpg_store_info = pd.read_csv(INPUT_DIR / 'hpg_store_info.csv')
air_store_info = pd.read_csv(INPUT_DIR / 'air_store_info.csv')
store_rel = pd.read_csv(INPUT_DIR / 'store_id_relation.csv')

hpg_res['visit_date'] = hpg_res['visit_datetime'].dt.normalize()
air_res['visit_date'] = air_res['visit_datetime'].dt.normalize()
hpg_res = hpg_res.merge(hpg_store_info[['hpg_store_id', 'hpg_area_name']], on='hpg_store_id', how='left')
hpg_res['city'] = hpg_res['hpg_area_name'].str.split(' ').str[:2].str.join(' ')
hpg_res['prefecture'] = hpg_res['hpg_area_name'].str.split(' ').str[0]
air_store_info['city'] = air_store_info['air_area_name'].str.split(' ').str[:2].str.join(' ')
air_store_info['prefecture'] = air_store_info['air_area_name'].str.split(' ').str[0]

hpg_city_daily = hpg_res.groupby(['city', 'visit_date']).agg(
    hpg_city_reserve_visitors=('reserve_visitors', 'sum'),
    hpg_city_reserve_count=('reserve_visitors', 'count')).reset_index()
hpg_pref_daily = hpg_res.groupby(['prefecture', 'visit_date']).agg(
    hpg_pref_reserve_visitors=('reserve_visitors', 'sum')).reset_index()
hpg_direct = hpg_res.merge(store_rel, on='hpg_store_id', how='inner')
hpg_store_daily = hpg_direct.groupby(['air_store_id', 'visit_date']).agg(
    hpg_store_reserve_visitors=('reserve_visitors', 'sum')).reset_index()
air_store_daily = air_res.groupby(['air_store_id', 'visit_date']).agg(
    air_reserve_visitors=('reserve_visitors', 'sum'),
    air_reserve_count=('reserve_visitors', 'count')).reset_index()

def add_reserve_features(df):
    df = df.copy()
    store_city = air_store_info.set_index('air_store_id')[['city', 'prefecture']]
    df = df.merge(store_city, left_on='air_store_id', right_index=True, how='left', suffixes=('', '_res'))
    df = df.merge(hpg_city_daily, on=['city', 'visit_date'], how='left')
    df = df.merge(hpg_pref_daily, on=['prefecture', 'visit_date'], how='left')
    df = df.merge(hpg_store_daily, on=['air_store_id', 'visit_date'], how='left')
    df = df.merge(air_store_daily, on=['air_store_id', 'visit_date'], how='left')
    df['total_reserve_visitors'] = df[['hpg_store_reserve_visitors', 'air_reserve_visitors']].sum(axis=1, min_count=1)
    reserve_cols = ['hpg_city_reserve_visitors', 'hpg_city_reserve_count',
                    'hpg_pref_reserve_visitors', 'hpg_store_reserve_visitors',
                    'air_reserve_visitors', 'air_reserve_count', 'total_reserve_visitors']
    for col in reserve_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df.drop(columns=['city', 'prefecture'], errors='ignore', inplace=True)
    for c in [x for x in df.columns if x.endswith('_res')]:
        df.drop(columns=[c], errors='ignore', inplace=True)
    return df

train_df = add_reserve_features(train_df)
valid_df = add_reserve_features(valid_df)

reserve_features = ['hpg_city_reserve_visitors', 'hpg_city_reserve_count',
                    'hpg_pref_reserve_visitors', 'hpg_store_reserve_visitors',
                    'air_reserve_visitors', 'air_reserve_count', 'total_reserve_visitors']
final_features = all_features + reserve_features
print(f'  Features: {len(final_features)}')

full_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values('visit_date').reset_index(drop=True)

# === LightGBM params ===
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.02, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_samples': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

# === XGBoost params ===
xgb_params = {
    'objective': 'reg:squarederror', 'eval_metric': 'rmse',
    'max_depth': 6, 'learning_rate': 0.02, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_weight': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': SEED, 'n_jobs': -1, 'verbosity': 0,
}

# === 2. CV comparison: LGB vs XGB vs Ensemble ===
print('\n=== 2. CV: LGB vs XGB vs Ensemble ===')
cv_lgb, cv_xgb, cv_ens = [], [], []
best_weights = []

for i, fold in enumerate(val_folds, 1):
    vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
    tr = full_df[full_df['visit_date'] < vs]
    va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
    if len(tr) == 0 or len(va) == 0: continue

    y_tr = np.log1p(tr['visitors'])
    y_va = np.log1p(va['visitors'])
    X_tr, X_va = tr[final_features], va[final_features]

    # LightGBM
    dtr_l = lgb.Dataset(X_tr, label=y_tr)
    dva_l = lgb.Dataset(X_va, label=y_va, reference=dtr_l)
    m_lgb = lgb.train(lgb_params, dtr_l, num_boost_round=2000, valid_sets=[dva_l],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p_lgb = np.expm1(m_lgb.predict(X_va))
    s_lgb = rmsle(va['visitors'], p_lgb)
    cv_lgb.append(s_lgb)

    # XGBoost
    dtr_x = xgb.DMatrix(X_tr, label=y_tr)
    dva_x = xgb.DMatrix(X_va, label=y_va)
    m_xgb = xgb.train(xgb_params, dtr_x, num_boost_round=2000, evals=[(dva_x, 'valid')],
                       early_stopping_rounds=100, verbose_eval=False)
    p_xgb = np.expm1(m_xgb.predict(dva_x))
    s_xgb = rmsle(va['visitors'], p_xgb)
    cv_xgb.append(s_xgb)

    # Grid search best weight
    best_w, best_s = 0.5, float('inf')
    for w in np.arange(0.1, 1.0, 0.05):
        p_ens = np.clip(p_lgb * w + p_xgb * (1 - w), 1, None)
        s_ens = rmsle(va['visitors'], p_ens)
        if s_ens < best_s:
            best_w, best_s = w, s_ens
    cv_ens.append(best_s)
    best_weights.append(best_w)

    print(f'  Fold {i}: LGB={s_lgb:.5f}, XGB={s_xgb:.5f}, Ens={best_s:.5f} (w_lgb={best_w:.2f})')

print(f'\n  LGB  Mean: {np.mean(cv_lgb):.5f} +/- {np.std(cv_lgb):.5f}')
print(f'  XGB  Mean: {np.mean(cv_xgb):.5f} +/- {np.std(cv_xgb):.5f}')
print(f'  ENS  Mean: {np.mean(cv_ens):.5f} +/- {np.std(cv_ens):.5f}')
mean_w = np.mean(best_weights)
print(f'  Mean weight (LGB): {mean_w:.2f}')

ens_improvement = np.mean(cv_lgb) - np.mean(cv_ens)
print(f'  Ensemble improvement vs LGB: {ens_improvement:+.5f}')

# === 3. Single split ===
print('\n=== 3. Single split comparison ===')
y_train = np.log1p(train_df['visitors'])
y_valid = np.log1p(valid_df['visitors'])

dtrain_l = lgb.Dataset(train_df[final_features], label=y_train)
dvalid_l = lgb.Dataset(valid_df[final_features], label=y_valid, reference=dtrain_l)
model_lgb = lgb.train(lgb_params, dtrain_l, num_boost_round=2000, valid_sets=[dvalid_l],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
pred_lgb = np.expm1(model_lgb.predict(valid_df[final_features]))

dtrain_x = xgb.DMatrix(train_df[final_features], label=y_train)
dvalid_x = xgb.DMatrix(valid_df[final_features], label=y_valid)
model_xgb = xgb.train(xgb_params, dtrain_x, num_boost_round=2000, evals=[(dvalid_x, 'valid')],
                       early_stopping_rounds=100, verbose_eval=False)
pred_xgb = np.expm1(model_xgb.predict(dvalid_x))

s_lgb = rmsle(valid_df['visitors'], pred_lgb)
s_xgb = rmsle(valid_df['visitors'], pred_xgb)

# Optimal weight on single split
best_w_single, best_s_single = 0.5, float('inf')
for w in np.arange(0.1, 1.0, 0.01):
    p = np.clip(pred_lgb * w + pred_xgb * (1 - w), 1, None)
    s = rmsle(valid_df['visitors'], p)
    if s < best_s_single:
        best_w_single, best_s_single = w, s

print(f'  LGB: {s_lgb:.5f} (iter={model_lgb.best_iteration})')
print(f'  XGB: {s_xgb:.5f} (iter={model_xgb.best_iteration})')
print(f'  ENS: {best_s_single:.5f} (w_lgb={best_w_single:.2f})')

# Use CV-based weight for final prediction
w_lgb = round(mean_w, 2)
print(f'\n  Using CV mean weight: w_lgb={w_lgb}')

# === 4. Final submission ===
print('\n=== 4. Final submission ===')
full_df_concat = pd.concat([train_df, valid_df], ignore_index=True)
y_all = np.log1p(full_df_concat['visitors'])

# LightGBM final
d_all_l = lgb.Dataset(full_df_concat[final_features], label=y_all)
model_final_lgb = lgb.train(lgb_params, d_all_l, num_boost_round=model_lgb.best_iteration)

# XGBoost final
d_all_x = xgb.DMatrix(full_df_concat[final_features], label=y_all)
model_final_xgb = xgb.train(xgb_params, d_all_x, num_boost_round=model_xgb.best_iteration)

# Build test features
sample_sub = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
sample_sub['air_store_id'] = sample_sub['id'].str[:-11]
sample_sub['visit_date'] = pd.to_datetime(sample_sub['id'].str[-10:])

air_store = pd.read_csv(INPUT_DIR / 'air_store_info.csv')
date_info = pd.read_csv(INPUT_DIR / 'date_info.csv', parse_dates=['calendar_date'])
test_df = sample_sub[['air_store_id', 'visit_date']].copy()
test_df = test_df.merge(air_store, on='air_store_id', how='left')
test_df = test_df.merge(date_info.rename(columns={'calendar_date': 'visit_date'}), on='visit_date', how='left')
test_df['month'] = test_df['visit_date'].dt.month
test_df['day'] = test_df['visit_date'].dt.day
test_df['dow'] = test_df['visit_date'].dt.dayofweek
test_df['week'] = test_df['visit_date'].dt.isocalendar().week.astype(int)
test_df['is_weekend'] = (test_df['dow'] >= 5).astype(int)
test_df['is_holiday'] = test_df['holiday_flg'].fillna(0).astype(int)

genre_map = full_df_concat.groupby('air_genre_name')['genre_encoded'].first().to_dict()
area_map = full_df_concat.groupby('air_area_name')['area_encoded'].first().to_dict()
test_df['genre_encoded'] = test_df['air_genre_name'].map(genre_map).fillna(-1).astype(int)
test_df['area_encoded'] = test_df['air_area_name'].map(area_map).fillna(-1).astype(int)

test_df = test_df.merge(prev['store_stats'], on='air_store_id', how='left')
test_df = test_df.merge(prev['store_dow'], on=['air_store_id', 'dow'], how='left')
test_df = test_df.merge(prev['genre_stats'], on='air_genre_name', how='left')
for ln, ldf in prev['genre_lookups'].items():
    mk = [c for c in ldf.columns if c in test_df.columns and c != ldf.columns[-1]]
    nc = ldf.columns[-1]
    if nc in test_df.columns: test_df.drop(columns=[nc], inplace=True)
    test_df = test_df.merge(ldf, on=mk, how='left')
test_df['store_popularity_in_genre'] = test_df['store_mean'] / test_df['genre_mean']

grid_df = prev['grid_df']
rlc = [c for c in grid_df.columns if any(c.startswith(p) for p in [
    'rolling_mean_', 'rolling_std_', 'ewm_mean', 'lag_', 'open_ratio_', 'closed_streak', 'days_since_long_closure'])]
ld = grid_df['visit_date'].max()
frz = grid_df[grid_df['visit_date'] == ld][['air_store_id'] + rlc].copy()
test_df = test_df.merge(frz, on='air_store_id', how='left', suffixes=('', '_f'))
for col in rlc:
    if f'{col}_f' in test_df.columns:
        test_df[col] = test_df[f'{col}_f'].fillna(test_df.get(col, np.nan))
        test_df.drop(columns=[f'{col}_f'], inplace=True)

dh = date_info.set_index('calendar_date')['holiday_flg'].to_dict()
for shift in [1, 2, 3]:
    test_df[f'is_after_holiday_{shift}'] = test_df['visit_date'].apply(lambda d: dh.get(d - pd.Timedelta(days=shift), 0))
for w in [7, 14]:
    test_df[f'holiday_count_{w}'] = test_df['visit_date'].apply(lambda d: sum(dh.get(d - pd.Timedelta(days=i), 0) for i in range(w)))
def is_near_special(d):
    for sm, sd in [(1,1),(1,2),(1,3),(4,29),(5,3),(5,4),(5,5),(8,13),(8,14),(8,15),(12,25),(12,31)]:
        if abs((d - pd.Timestamp(d.year, sm, sd)).days) <= 3: return 1
    return 0
test_df['is_near_special_period'] = test_df['visit_date'].apply(is_near_special)

test_df = add_reserve_features(test_df)
for f in final_features:
    if f not in test_df.columns: test_df[f] = 0

X_test = test_df[final_features]
print(f'  Test NaN: {X_test.isna().mean().mean()*100:.1f}%')

# Predictions
test_lgb = np.expm1(model_final_lgb.predict(X_test))
test_xgb = np.expm1(model_final_xgb.predict(xgb.DMatrix(X_test)))
test_ens = np.clip(test_lgb * w_lgb + test_xgb * (1 - w_lgb), 1, None)

# Save ensemble submission
sub = sample_sub[['id']].copy()
sub['visitors'] = test_ens
sub.to_csv(OUTPUT_DIR / 'submission_ensemble.csv', index=False)

# Also save LGB-only for comparison
sub_lgb = sample_sub[['id']].copy()
sub_lgb['visitors'] = np.clip(test_lgb, 1, None)
sub_lgb.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

print(f'  LGB pred:  mean={test_lgb.mean():.2f}')
print(f'  XGB pred:  mean={test_xgb.mean():.2f}')
print(f'  ENS pred:  mean={test_ens.mean():.2f}, median={np.median(test_ens):.2f}, min={test_ens.min():.2f}, max={test_ens.max():.2f}')
print(f'  Correlation LGB-XGB: {np.corrcoef(test_lgb, test_xgb)[0,1]:.4f}')
print('  Done!')
