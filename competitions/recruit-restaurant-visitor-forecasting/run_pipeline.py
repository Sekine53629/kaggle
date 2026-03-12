"""03-1 LightGBM v4: Reserve features + Optuna pipeline"""
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

lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'num_leaves': 63, 'learning_rate': 0.02, 'subsample': 0.8,
    'colsample_bytree': 0.8, 'min_child_samples': 20,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

# === Build reserve features ===
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
all_features_v2 = all_features + reserve_features
print(f'  Features: {len(all_features)} -> {len(all_features_v2)}')

# === 2. CV: original ===
print('\n=== 2. CV: original features ===')
full_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values('visit_date').reset_index(drop=True)

cv_orig = []
for i, fold in enumerate(val_folds, 1):
    vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
    tr = full_df[full_df['visit_date'] < vs]
    va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
    if len(tr) == 0 or len(va) == 0: continue
    dtr = lgb.Dataset(tr[all_features], label=np.log1p(tr['visitors']))
    dva = lgb.Dataset(va[all_features], label=np.log1p(va['visitors']), reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    s = rmsle(va['visitors'], np.expm1(m.predict(va[all_features])))
    cv_orig.append(s)
    print(f'  Fold {i}: {s:.5f}')
print(f'  Mean: {np.mean(cv_orig):.5f} +/- {np.std(cv_orig):.5f}')

# === 3. CV: with reserve ===
print('\n=== 3. CV: with reserve features ===')
cv_v2 = []
for i, fold in enumerate(val_folds, 1):
    vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
    tr = full_df[full_df['visit_date'] < vs]
    va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
    if len(tr) == 0 or len(va) == 0: continue
    dtr = lgb.Dataset(tr[all_features_v2], label=np.log1p(tr['visitors']))
    dva = lgb.Dataset(va[all_features_v2], label=np.log1p(va['visitors']), reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    s = rmsle(va['visitors'], np.expm1(m.predict(va[all_features_v2])))
    cv_v2.append(s)
    diff = cv_orig[i-1] - s
    print(f'  Fold {i}: {s:.5f} (diff: {diff:+.5f})')

print(f'  Mean: {np.mean(cv_v2):.5f} +/- {np.std(cv_v2):.5f}')
imp = np.mean(cv_orig) - np.mean(cv_v2)
print(f'  Improvement: {imp:+.5f}')
USE_RESERVE = imp > 0
print(f'  Decision: {"USE RESERVE" if USE_RESERVE else "SKIP RESERVE"}')

# === 4. Single split ===
print('\n=== 4. Single split ===')
final_features = all_features_v2 if USE_RESERVE else all_features
dtrain = lgb.Dataset(train_df[final_features], label=np.log1p(train_df['visitors']))
dvalid = lgb.Dataset(valid_df[final_features], label=np.log1p(valid_df['visitors']), reference=dtrain)
model = lgb.train(lgb_params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
pred = np.expm1(model.predict(valid_df[final_features]))
score_single = rmsle(valid_df['visitors'], pred)
print(f'  RMSLE: {score_single:.5f} (iter={model.best_iteration})')

# === 4b. Optuna tuning (60 features) ===
print('\n=== 4b. Optuna tuning (60 features) ===')
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 50

def objective_lgb(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbose': -1,
        'random_state': SEED, 'n_jobs': -1,
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    fold_scores = []
    for fold in val_folds:
        vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
        tr = full_df[full_df['visit_date'] < vs]
        va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
        if len(tr) == 0 or len(va) == 0: continue
        dtr = lgb.Dataset(tr[final_features], label=np.log1p(tr['visitors']))
        dva = lgb.Dataset(va[final_features], label=np.log1p(va['visitors']), reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        p = np.expm1(m.predict(va[final_features]))
        fold_scores.append(rmsle(va['visitors'], p))
    return np.mean(fold_scores) if fold_scores else float('inf')

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective_lgb, n_trials=N_TRIALS)

optuna_cv = study.best_value
default_cv = np.mean(cv_v2)
optuna_improvement = default_cv - optuna_cv
print(f'  Default CV:  {default_cv:.5f}')
print(f'  Optuna CV:   {optuna_cv:.5f}')
print(f'  Improvement: {optuna_improvement:+.5f}')

# Use Optuna params if improvement is meaningful (> 5% of CV std)
cv_std = np.std(cv_v2)
USE_OPTUNA = optuna_improvement > cv_std * 0.05
if USE_OPTUNA:
    best_params = {**study.best_params, 'objective': 'regression', 'metric': 'rmse',
                   'verbose': -1, 'random_state': SEED, 'n_jobs': -1}
    # Re-train single split with Optuna params
    dtrain_o = lgb.Dataset(train_df[final_features], label=np.log1p(train_df['visitors']))
    dvalid_o = lgb.Dataset(valid_df[final_features], label=np.log1p(valid_df['visitors']), reference=dtrain_o)
    model_o = lgb.train(best_params, dtrain_o, num_boost_round=2000, valid_sets=[dvalid_o],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred_o = np.expm1(model_o.predict(valid_df[final_features]))
    score_o = rmsle(valid_df['visitors'], pred_o)
    print(f'  Optuna Single: {score_o:.5f} (default: {score_single:.5f})')
    if score_o < score_single:
        model = model_o
        score_single = score_o
        pred = pred_o
        lgb_params = best_params
        print(f'  -> Using Optuna params (iter={model.best_iteration})')
    else:
        print(f'  -> Optuna worse on single split, keeping default')
        USE_OPTUNA = False
else:
    print(f'  -> Improvement {optuna_improvement:.5f} < threshold {cv_std*0.05:.5f}, keeping default')

for k, v in study.best_params.items():
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

# === 5. Feature importance (v2 model) ===
print('\n=== 5. Feature importance (Top 20) ===')
imp_gain = model.feature_importance(importance_type='gain')
imp_df = pd.DataFrame({'feature': final_features, 'gain': imp_gain}).sort_values('gain', ascending=False)
imp_df['pct'] = imp_df['gain'] / imp_df['gain'].sum() * 100
imp_df['cum_pct'] = imp_df['pct'].cumsum()
for _, r in imp_df.head(20).iterrows():
    marker = ' [RESERVE]' if r['feature'] in reserve_features else ''
    print(f'  {r["feature"]:40s} {r["pct"]:5.1f}% (cum {r["cum_pct"]:5.1f}%){marker}')
top2_share = imp_df.head(2)['pct'].sum()
print(f'  Top2 share: {top2_share:.1f}%')
reserve_share = imp_df[imp_df['feature'].isin(reserve_features)]['pct'].sum()
print(f'  Reserve features total share: {reserve_share:.1f}%')

# === 6. Smearing correction (Duan's estimator) ===
print('\n=== 6. Smearing correction ===')
train_mean = full_df['visitors'].mean()
train_median = full_df['visitors'].median()

# Compute OOF residuals in log space for smearing factor
oof_residuals = []
for i, fold in enumerate(val_folds, 1):
    vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
    tr = full_df[full_df['visit_date'] < vs]
    va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
    if len(tr) == 0 or len(va) == 0: continue
    dtr = lgb.Dataset(tr[final_features], label=np.log1p(tr['visitors']))
    dva = lgb.Dataset(va[final_features], label=np.log1p(va['visitors']), reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    log_pred = m.predict(va[final_features])
    log_actual = np.log1p(va['visitors'].values)
    residuals = log_actual - log_pred
    oof_residuals.extend(residuals)

oof_residuals = np.array(oof_residuals)
# Duan's smearing: E[exp(residual)]
smearing_factor = np.mean(np.exp(oof_residuals))
print(f'  Duan smearing factor: {smearing_factor:.4f}')
print(f'  Residual mean: {oof_residuals.mean():.4f}, std: {oof_residuals.std():.4f}')

# Validate smearing on single split
pred_valid_log = model.predict(valid_df[final_features])
pred_no_smear = np.clip(np.expm1(pred_valid_log), 1, None)
pred_smeared = np.clip(np.expm1(pred_valid_log) * smearing_factor, 1, None)
score_no_smear = rmsle(valid_df['visitors'], pred_no_smear)
score_smeared = rmsle(valid_df['visitors'], pred_smeared)
print(f'  Valid RMSLE (no smear):  {score_no_smear:.5f}  mean={pred_no_smear.mean():.1f}')
print(f'  Valid RMSLE (smeared):   {score_smeared:.5f}  mean={pred_smeared.mean():.1f}')
print(f'  Actual mean: {valid_df["visitors"].mean():.1f}')

# Also test simple multiplicative correction from validation
mult_factor = valid_df['visitors'].mean() / pred_no_smear.mean()
pred_mult = np.clip(pred_no_smear * mult_factor, 1, None)
score_mult = rmsle(valid_df['visitors'], pred_mult)
print(f'  Valid RMSLE (mult={mult_factor:.4f}): {score_mult:.5f}  mean={pred_mult.mean():.1f}')

# CV evaluation of smearing
print('\n  CV with smearing:')
cv_smeared = []
for i, fold in enumerate(val_folds, 1):
    vs, ve = pd.Timestamp(fold['val_start']), pd.Timestamp(fold['val_end'])
    tr = full_df[full_df['visit_date'] < vs]
    va = full_df[(full_df['visit_date'] >= vs) & (full_df['visit_date'] <= ve)]
    if len(tr) == 0 or len(va) == 0: continue
    dtr = lgb.Dataset(tr[final_features], label=np.log1p(tr['visitors']))
    dva = lgb.Dataset(va[final_features], label=np.log1p(va['visitors']), reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    p_log = m.predict(va[final_features])
    p_smeared = np.clip(np.expm1(p_log) * smearing_factor, 1, None)
    s = rmsle(va['visitors'], p_smeared)
    cv_smeared.append(s)
    print(f'    Fold {i}: {s:.5f}')
print(f'    Mean: {np.mean(cv_smeared):.5f} +/- {np.std(cv_smeared):.5f}')
print(f'    vs no-smear CV: {np.mean(cv_v2):.5f}')
cv_smear_improvement = np.mean(cv_v2) - np.mean(cv_smeared)
print(f'    Improvement: {cv_smear_improvement:+.5f}')
USE_SMEARING = cv_smear_improvement > 0

# === 7. Final submission ===
print('\n=== 7. Final submission ===')
full_df = pd.concat([train_df, valid_df], ignore_index=True)
d_all = lgb.Dataset(full_df[final_features], label=np.log1p(full_df['visitors']))
model_final = lgb.train(lgb_params, d_all, num_boost_round=model.best_iteration)

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

genre_map = full_df.groupby('air_genre_name')['genre_encoded'].first().to_dict()
area_map = full_df.groupby('air_area_name')['area_encoded'].first().to_dict()
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

if USE_RESERVE:
    test_df = add_reserve_features(test_df)

for f in final_features:
    if f not in test_df.columns: test_df[f] = 0

X_test = test_df[final_features]
print(f'  Test NaN: {X_test.isna().mean().mean()*100:.1f}%')

test_pred_raw = np.clip(np.expm1(model_final.predict(X_test)), 1, None)
if USE_SMEARING:
    test_pred = np.clip(test_pred_raw * smearing_factor, 1, None)
    print(f'  Smearing applied: factor={smearing_factor:.4f}')
else:
    test_pred = test_pred_raw
    print(f'  Smearing skipped (no CV improvement)')

# Save both versions
sub = sample_sub[['id']].copy()
sub['visitors'] = test_pred
sub.to_csv(OUTPUT_DIR / 'submission.csv', index=False)

sub_raw = sample_sub[['id']].copy()
sub_raw['visitors'] = test_pred_raw
sub_raw.to_csv(OUTPUT_DIR / 'submission_no_smear.csv', index=False)

print(f'  Prediction: mean={test_pred.mean():.2f}, median={np.median(test_pred):.2f}, min={test_pred.min():.2f}, max={test_pred.max():.2f}')
print(f'  Raw pred:   mean={test_pred_raw.mean():.2f}')
print(f'  Train:      mean={train_mean:.2f}, median={train_median:.2f}')
print(f'  Bias:       {(test_pred.mean() - train_mean)/train_mean*100:+.1f}% vs train mean')
print('  Done!')
