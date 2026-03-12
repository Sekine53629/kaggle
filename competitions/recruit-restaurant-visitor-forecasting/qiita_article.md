# 飲食店の来客数予測をデータ分析で解く — Kaggleコンペティションへの挑戦

## はじめに

本記事では、Kaggleコンペティション「[Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting)」のデータを用いて、**日本国内の飲食店における来客数予測**に取り組んだ分析結果をまとめます。

データの探索的分析（EDA）から特徴量設計、複数モデルの構築・比較、アンサンブルまで、一連のデータ分析プロセスを通じて得られた知見を共有します。

---

## 解決したい社会課題

飲食店にとって、**来客数の予測精度**は経営の根幹に関わる課題です。

- **食材の仕入れ**: 過剰仕入れは廃棄ロスに、過少仕入れは品切れ・機会損失につながる
- **人員配置**: 適切なシフト計画ができなければ、人件費の無駄や人手不足によるサービス品質低下が発生する
- **売上予測**: 資金繰りや設備投資の判断に直結する

農林水産省の調査によると、日本の食品ロスは年間約472万トン（2022年度推計）であり、外食産業はその主要な発生源の一つです。**来客数を正確に予測できれば、仕入れ量の最適化を通じて食品ロスの削減に貢献できます**。

本分析では、リクルートが提供するレストラン予約・来客データをもとに、機械学習モデルによる来客数予測の精度をどこまで高められるかを検証します。

---

## 分析するデータ

リクルートが運営する2つの飲食店プラットフォームのデータを使用します。

- **AirREGI（エアレジ）**: POSレジアプリの来客データ・予約データ
- **Hot Pepper Gourmet**: グルメサイトの予約データ

| データ | 件数 | 内容 |
|--------|------|------|
| air_visit_data | 252,108件 | 来客実績（829店舗） |
| air_reserve | 92,378件 | Air経由の予約データ |
| hpg_reserve | 2,000,320件 | HPG経由の予約データ |
| air_store_info | 829件 | 店舗情報（ジャンル・地域・座標） |
| hpg_store_info | 4,690件 | HPG店舗情報 |
| store_id_relation | 150件 | Air-HPG店舗の紐づけ |
| date_info | 517日分 | 祝日フラグ付きカレンダー |

- **学習期間**: 2016年1月〜2017年4月
- **予測対象期間**: 2017年4月23日〜5月31日（39日間）
- **評価指標**: **RMSLE**（Root Mean Squared Logarithmic Error）
  - 過小予測に対してより大きなペナルティを与える指標
  - 飲食店にとって「足りない」方が「余る」より深刻であるという業界特性に合致

---

## 実行環境

| 項目 | 詳細 |
|------|------|
| PC | Microsoft Surface Laptop 7th Edition |
| プロセッサ | Snapdragon(R) X 10-core X1P64100 @ 3.40 GHz |
| 開発環境 | Python 3.13 / Jupyter Notebook |

### 使用ライブラリ

| カテゴリ | ライブラリ |
|----------|-----------|
| データ処理 | pandas, numpy, pickle |
| 可視化 | matplotlib, seaborn |
| 機械学習 | scikit-learn, lightgbm, xgboost, catboost |
| ハイパーパラメータ最適化 | optuna |
| 統計分析 | statsmodels, scipy |

---

## 分析の流れ

```
Step 1: 探索的データ分析（EDA）
    ↓ データの全体像・傾向・課題を把握
Step 2: 特徴量設計
    ↓ 53個の特徴量を段階的に構築・検証
Step 3: モデル構築（4モデル）
    ↓ LightGBM / XGBoost / RandomForest / CatBoost
Step 4: モデル比較・アンサンブル
    ↓ 横断比較・重み付きアンサンブル
Step 5: 最終結果のまとめ
```

---

## 分析の過程

### Step 1: 探索的データ分析（EDA）

#### 1-1. データの読み込みとプロファイリング

まず、7つのデータセットを読み込み、全体像を把握します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

INPUT_DIR = Path('../../input')

air_visit = pd.read_csv(INPUT_DIR / 'air_visit_data.csv', parse_dates=['visit_date'])
air_reserve = pd.read_csv(INPUT_DIR / 'air_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime'])
air_store = pd.read_csv(INPUT_DIR / 'air_store_info.csv')
hpg_reserve = pd.read_csv(INPUT_DIR / 'hpg_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime'])
hpg_store = pd.read_csv(INPUT_DIR / 'hpg_store_info.csv')
store_relation = pd.read_csv(INPUT_DIR / 'store_id_relation.csv')
date_info = pd.read_csv(INPUT_DIR / 'date_info.csv', parse_dates=['calendar_date'])

print(f'来客データ: {air_visit.shape[0]:,}件, {air_visit["air_store_id"].nunique()}店舗')
print(f'期間: {air_visit["visit_date"].min()} 〜 {air_visit["visit_date"].max()}')
```

```
来客データ: 252,108件, 829店舗
期間: 2016-01-01 〜 2017-04-22
```

#### 1-2. 目的変数の分布確認

来客数（visitors）の基本統計量を確認しました。

| 統計量 | 値 |
|--------|-----|
| 平均 | 21.0人 |
| 中央値 | 17人 |
| 標準偏差 | 16.8人 |
| 最大値 | 877人 |
| 歪度 | 3.31 |

分布は強い右裾（正の歪度）を持つため、**log1p変換**を適用しました。これにより分布が正規分布に近づき、RMSLE評価指標とも整合します。

```python
# log1p変換で正規分布に近づける
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(air_visit['visitors'], bins=100, alpha=0.7)
axes[0].set_title('来客数の分布（元データ）')

axes[1].hist(np.log1p(air_visit['visitors']), bins=100, alpha=0.7, color='orange')
axes[1].set_title('来客数の分布（log1p変換後）')
plt.tight_layout()
plt.show()
```

#### 1-3. 曜日別パターンの分析

```python
air_visit['dow'] = air_visit['visit_date'].dt.dayofweek
dow_names = ['月', '火', '水', '木', '金', '土', '日']

dow_stats = air_visit.groupby('dow')['visitors'].agg(['mean', 'median', 'std'])

# 平日と週末で色を分けて可視化
colors = ['#4C72B0'] * 5 + ['#DD8452'] * 2  # 平日=青, 週末=オレンジ
plt.bar(range(7), dow_stats['mean'], yerr=dow_stats['std'], color=colors, capsize=3, alpha=0.8)
plt.xticks(range(7), dow_names)
plt.ylabel('平均来客数')
plt.title('曜日別の平均来客数')
plt.show()
```

**金曜日・土曜日がピーク**であることが明確に確認されました。この曜日パターンは全ジャンルに共通しますが、強度はジャンルにより異なります。

- **居酒屋**: 金曜のピークが顕著（平日の約1.5倍）
- **カフェ・スイーツ**: 週末効果が比較的弱い

さらに、曜日×月のヒートマップで季節変動も確認しました。

```python
# 曜日 × 月 のヒートマップ
air_visit['month'] = air_visit['visit_date'].dt.month
dow_month = air_visit.groupby(['dow', 'month'])['visitors'].mean().unstack()

sns.heatmap(dow_month, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=[f'{m}月' for m in range(1, 13)],
            yticklabels=dow_names)
plt.title('曜日×月の平均来客数')
plt.show()
```

**12月（忘年会シーズン）の金曜・土曜が年間最高**であることが視覚的にわかります。

#### 1-4. 店舗ライフサイクルと定休日パターンの分析

各店舗の営業開始日・終了日・欠損日数を算出し、定休日パターンを特定しました。

```python
# 店舗ごとの初日・最終日・欠損日数
store_lifecycle = air_visit.groupby('air_store_id')['visit_date'].agg(
    first_date='min', last_date='max', n_records='count'
)
store_lifecycle['expected_days'] = (store_lifecycle['last_date'] - store_lifecycle['first_date']).dt.days + 1
store_lifecycle['missing_days'] = store_lifecycle['expected_days'] - store_lifecycle['n_records']
store_lifecycle['missing_rate'] = store_lifecycle['missing_days'] / store_lifecycle['expected_days']

print(f'欠損日数: 平均{store_lifecycle["missing_days"].mean():.1f}日, '
      f'最大{store_lifecycle["missing_days"].max()}日')
print(f'欠損率20%超の店舗: {(store_lifecycle["missing_rate"] > 0.2).sum()}店')
```

```
欠損日数: 平均53.3日, 最大351日
欠損率20%超の店舗: 210店
```

定休日パターンの検出では、各店舗の欠損曜日を集計し、特定曜日に40%以上集中している場合を「定期休業日あり」と判定しました。

```python
def analyze_missing_pattern(store_id, visit_df):
    """店舗の欠損パターンを分類: 定休日 vs 不規則休業"""
    store_dates = set(visit_df[visit_df['air_store_id'] == store_id]['visit_date'])
    first, last = min(store_dates), max(store_dates)
    active_range = pd.date_range(first, last, freq='D')
    missing = [d for d in active_range if d not in store_dates]

    if not missing:
        return pd.Series({'regular_closure_dow': -1, 'total_missing': 0})

    missing_dow = pd.Series([d.dayofweek for d in missing])
    dow_counts = missing_dow.value_counts()
    most_common_dow = dow_counts.index[0]
    most_common_pct = dow_counts.iloc[0] / len(missing)

    return pd.Series({
        'regular_closure_dow': most_common_dow if most_common_pct > 0.4 else -1,
        'total_missing': len(missing)
    })
```

829店舗中**537店舗**に週1回の定期休業日が確認されました。

| 定休曜日 | 店舗数 |
|---------|-----:|
| 日曜 | 225 |
| 月曜 | 157 |
| 火曜 | 64 |

#### 1-5. 時系列のトレンド分解（STL）

時系列データをトレンド・季節性・残差に分解し、構造を把握しました。

```python
from statsmodels.tsa.seasonal import STL

# 日次平均来客数を算出し、STL分解
daily_mean = air_visit.groupby('visit_date')['visitors'].mean()
daily_mean = daily_mean.asfreq('D').interpolate()

stl = STL(daily_mean, period=7, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(18, 14))
result.observed.plot(ax=axes[0], title='Observed（実測値）')
result.trend.plot(ax=axes[1], title='Trend（トレンド）', color='orange')
result.seasonal.plot(ax=axes[2], title='Seasonal（週次季節性）', color='green')
result.resid.plot(ax=axes[3], title='Residual（残差）', color='red')
plt.tight_layout()
plt.show()
```

- **トレンド**: 緩やかな上昇傾向
- **季節性**: 安定した週次パターン（金土がピーク）
- **残差**: 正月・GW・お盆などの特殊期間で大きな残差が発生

#### 1-6. 祝日・特殊期間の影響分析

祝日当日だけでなく、**前日・翌日の波及効果**も分析しました。

```python
# 祝日の前日・翌日フラグを作成
date_info['holiday_tomorrow'] = date_info['holiday_flg'].shift(-1).fillna(0).astype(int)
date_info['holiday_yesterday'] = date_info['holiday_flg'].shift(1).fillna(0).astype(int)
date_info['is_before_holiday'] = (
    (date_info['holiday_flg'] == 0) & (date_info['holiday_tomorrow'] == 1)
).astype(int)
date_info['is_after_holiday'] = (
    (date_info['holiday_flg'] == 0) & (date_info['holiday_yesterday'] == 1)
).astype(int)

# カテゴリ別の来客数を比較
categories = {
    '通常平日': (visit_ext['holiday_flg'] == 0) & (visit_ext['is_before_holiday'] == 0)
               & (visit_ext['is_after_holiday'] == 0),
    '祝日前日': visit_ext['is_before_holiday'] == 1,
    '祝日当日': visit_ext['holiday_flg'] == 1,
    '祝日翌日': visit_ext['is_after_holiday'] == 1,
}

for name, mask in categories.items():
    mean_v = visit_ext.loc[mask, 'visitors'].mean()
    print(f'{name}: 平均 {mean_v:.1f}人')
```

| 期間 | 平均来客数 | 通常比 | 営業店舗の変動 |
|------|-----------|--------|--------------|
| 正月 2017 | 26.0人 | +27.8% | -62.7%（営業店舗が大幅減少） |
| GW 2016 | 26.9人 | +30.6% | -53.3% |
| クリスマス 2016 | 25.9人 | +25.7% | -29.7% |
| 年末 2016 | 27.2人 | +33.5% | -6.7% |

**重要な発見**: 祝日には多くの店舗が休業するため、営業を続ける店舗に来客が集中する「**サバイバー効果**」が存在します。また、**祝日前日は来客数が増加**する傾向（前夜飲み効果）も確認されました。

年末年始の来客増加率はジャンルにより大きく異なります。

| ジャンル | 年末来客増加率 |
|----------|-------------|
| 焼肉・韓国料理 | 1.60倍 |
| 居酒屋 | 1.48倍 |
| カフェ・スイーツ | 1.04倍（ほぼ変化なし） |

#### 1-7. 予約データのカバレッジ分析

```python
# Air予約データと来客データを結合して、カバレッジを確認
air_reserve['visit_date'] = air_reserve['visit_datetime'].dt.date
reserve_daily = air_reserve.groupby(['air_store_id', 'visit_date'])['reserve_visitors'].sum().reset_index()
merged = air_visit.merge(reserve_daily, on=['air_store_id', 'visit_date'], how='left')

coverage = merged['reserve_visitors'].notna().mean()
print(f'予約データのカバレッジ: {coverage:.1%}')

# カバー率（予約あり行のみ）
has_reserve = merged[merged['reserve_visitors'].notna()]
has_reserve['cover_ratio'] = has_reserve['reserve_visitors'] / has_reserve['visitors']
print(f'予約カバー率（中央値）: {has_reserve["cover_ratio"].median():.1%}')
```

- 予約データが紐づく来客実績: 全体のわずか**11.1%**
- 予約がある場合のカバー率（中央値）: 実来客数の**51%**
- 予約数と来客数の相関: **0.461**（中程度の正の相関）

#### 1-8. バリデーション戦略の設計

テスト期間（4月23日〜5月31日）にはゴールデンウィークが含まれるため、GWを含むFoldを必ず設ける**5-fold時系列CV**を設計しました。

```python
# テスト期間と同じ39日間の検証窓を5つ設計
val_folds = [
    {'name': 'Fold 1 (GW 2016含む)',
     'train_end': '2016-04-22', 'val_start': '2016-04-23', 'val_end': '2016-05-31'},
    {'name': 'Fold 2 (夏季)',
     'train_end': '2016-07-15', 'val_start': '2016-07-16', 'val_end': '2016-08-23'},
    {'name': 'Fold 3 (秋季)',
     'train_end': '2016-10-14', 'val_start': '2016-10-15', 'val_end': '2016-11-22'},
    {'name': 'Fold 4 (年末年始)',
     'train_end': '2016-12-15', 'val_start': '2016-12-16', 'val_end': '2017-01-23'},
    {'name': 'Fold 5 (テスト直前期)',
     'train_end': '2017-03-14', 'val_start': '2017-03-15', 'val_end': '2017-04-22'},
]
```

| Fold | 検証期間 | 特徴 |
|------|---------|------|
| 1 | 2016-04-23 〜 2016-05-31 | **GW含む（最重要）** |
| 2 | 2016-07-16 〜 2016-08-23 | 夏季 |
| 3 | 2016-10-15 〜 2016-11-22 | 秋季 |
| 4 | 2016-12-16 〜 2017-01-23 | 年末年始 |
| 5 | 2017-03-15 〜 2017-04-22 | テスト直前期 |

---

### Step 2: 特徴量設計

EDAで得られた知見をもとに、53個の特徴量を**3層構造**で段階的に設計しました。

#### 2-1. 時間特徴量と店舗統計量

まず、基本的な時間特徴量と、店舗ごとの過去統計量を作成します。店舗統計量はリーク防止のため、**学習データのみ**から算出します。

```python
# 時間特徴量の作成
def add_time_features(df):
    df = df.copy()
    df['month'] = df['visit_date'].dt.month
    df['day'] = df['visit_date'].dt.day
    df['dow'] = df['visit_date'].dt.dayofweek
    df['week'] = df['visit_date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_holiday'] = df['holiday_flg'].fillna(0).astype(int)
    return df

# 店舗統計量（リーク防止: 学習データのみから算出）
store_stats = train[train['visit_date'] < VALID_START].groupby('air_store_id')['visitors'].agg(
    ['mean', 'median', 'std', 'min', 'max', 'count']
).reset_index()
store_stats.columns = ['air_store_id', 'store_mean', 'store_median', 'store_std',
                        'store_min', 'store_max', 'store_count']

# 店舗×曜日統計量（最も重要な特徴量）
store_dow = train[train['visit_date'] < VALID_START].groupby(
    ['air_store_id', 'dow']
)['visitors'].agg(['mean', 'median']).reset_index()
store_dow.columns = ['air_store_id', 'dow', 'store_dow_mean', 'store_dow_median']
```

#### 2-2. Rolling/ラグ特徴量（3層アーキテクチャ）

EDAで発見した「週次周期」「定休日パターン」「祝日波及効果」を反映した3層構造の特徴量を構築しました。

```python
# === Layer 1: Rolling統計量とラグ特徴量 ===
# shift(1)で前日までのデータのみ使用（リーク防止）

windows = [7, 14, 21, 35, 63]
for w in windows:
    grouped = grid_df.groupby('air_store_id')['visitors']
    grid_df[f'rolling_mean_{w}'] = grouped.transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
    )
    grid_df[f'rolling_std_{w}'] = grouped.transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).std()
    )

# 指数加重移動平均（直近のデータにより大きな重みを付与）
grid_df['ewm_mean'] = grid_df.groupby('air_store_id')['visitors'].transform(
    lambda x: x.shift(1).ewm(span=14, min_periods=1).mean()
)

# ラグ特徴量（1日前、1週間前、…）
for lag in [1, 7, 14, 21, 28, 35]:
    grid_df[f'lag_{lag}'] = grid_df.groupby('air_store_id')['visitors'].shift(lag)
```

```python
# === Layer 2: 休業パターン特徴量 ===
# 営業率（直近N日間で何割営業していたか）
for w in [7, 14, 28]:
    grid_df[f'open_ratio_{w}'] = grid_df.groupby('air_store_id')['is_open'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
    )

# 連続休業日数
def calc_closed_streak(s):
    result = np.zeros(len(s))
    streak = 0
    for i in range(len(s)):
        if i == 0:
            result[i] = 0
        elif s.iloc[i - 1] == 0:  # 前日が休業
            streak += 1
            result[i] = streak
        else:
            streak = 0
            result[i] = 0
    return pd.Series(result, index=s.index)

grid_df['closed_streak'] = grid_df.groupby('air_store_id')['is_open'].transform(
    calc_closed_streak
)
```

```python
# === Layer 3: 祝日前後特徴量 ===
# N日前が祝日だったか
for n in [1, 2, 3]:
    grid_df[f'is_after_holiday_{n}'] = grid_df.groupby('air_store_id')['holiday_flg'].transform(
        lambda x: x.fillna(0).shift(n)
    ).astype(float)

# 直近N日間の祝日数
for w in [7, 14]:
    grid_df[f'holiday_count_{w}'] = grid_df.groupby('air_store_id')['holiday_flg'].transform(
        lambda x: x.fillna(0).shift(1).rolling(w, min_periods=1).sum()
    )
```

#### 2-3. 特徴量カテゴリ一覧

| カテゴリ | 特徴量数 | 主な特徴量 |
|----------|---------|-----------|
| 時間特徴量 | 6 | dow, month, day, week, is_weekend, is_holiday |
| 店舗属性 | 4 | genre_encoded, area_encoded, latitude, longitude |
| 店舗統計量 | 8 | store_mean, store_median, store_dow_mean, store_dow_median 等 |
| Rolling統計量 | 11 | rolling_mean_7/14/21/35/63, rolling_std, ewm_mean |
| ラグ特徴量 | 6 | lag_1, lag_7, lag_14, lag_21, lag_28, lag_35 |
| 休業パターン | 5 | open_ratio_7/14/28, closed_streak, days_since_long_closure |
| 祝日前後 | 6 | is_after_holiday_1/2/3, holiday_count_7/14, is_near_special_period |
| ジャンル関連 | 8 | genre_dow_mean, genre統計量 等 |

#### 2-4. 段階的な効果検証

特徴量を段階的に追加し、各段階でのCV RMSLEを計測しました。

```python
# Step 1: 時間特徴量のみ
time_features = ['month', 'day', 'dow', 'week', 'is_weekend', 'is_holiday']
lgb_data = lgb.Dataset(train_t[time_features], label=np.log1p(train_t['visitors']))
model_time = lgb.train(
    {'objective': 'regression', 'metric': 'rmse', 'verbose': -1},
    lgb_data, num_boost_round=300
)
pred = np.expm1(model_time.predict(valid_t[time_features]))
score_time = rmsle(valid_t['visitors'], pred)
print(f'時間特徴量のみ: RMSLE = {score_time:.5f}')

# 以降、カテゴリ追加ごとに同様に評価...
```

| ステップ | 特徴量数 | CV RMSLE | 改善幅 |
|----------|---------|----------|--------|
| 時間特徴量のみ | 6 | 0.80146 | — |
| ＋店舗統計量 | 20 | 0.56541 | **-0.236** |
| ＋ジャンル交互作用 | 25 | 0.56433 | -0.001 |
| ＋Rolling/ラグ | 42 | 0.54698 | -0.017 |
| ＋休業/祝日パターン | 53 | 0.54520 | -0.002 |

**最大の改善は「店舗統計量」の追加**（-0.236）で、各店舗の曜日別来客中央値がモデルのベースラインとして極めて有効であることがわかります。

#### 2-5. 学習期間・NaN戦略・Rolling構成の最適化

下流のモデル構築に影響する3つの設計選択を、CVベースで比較しました。

```python
# 学習期間の比較（5つの候補を評価）
train_periods = [
    ('直近15ヶ月', 456),
    ('直近12ヶ月', 365),
    ('直近9ヶ月',  274),
    ('直近6ヶ月',  183),
    ('直近3ヶ月',  91),
]

for label, period_days in train_periods:
    cv_mean, cv_std, cv_scores, _ = evaluate_with_tscv(
        df, grid_df, all_features, lgb_base_params,
        val_folds, train_period_days=period_days
    )
    print(f'{label}: CV={cv_mean:.5f} ± {cv_std:.5f}')
```

| 学習期間 | CV RMSLE | CV標準偏差 |
|---------|---------|----------|
| **直近15ヶ月** | **0.54520** | 0.02648 |
| 直近12ヶ月 | 0.54518 | 0.02651 |
| 直近9ヶ月 | 0.54742 | 0.02717 |
| 直近6ヶ月 | 0.54960 | 0.02481 |
| 直近3ヶ月 | 0.56269 | 0.02388 |

```python
# NaN戦略の比較
nan_strategies = {
    'NaN(デフォルト)': None,         # LightGBM/XGBoost/CatBoostはNaNをネイティブ処理
    '-999埋め': lambda X: X.fillna(-999),
    '0埋め': lambda X: X.fillna(0),
    '中央値埋め': lambda X: X.fillna(X.median()),
}

for strategy_name, fill_fn in nan_strategies.items():
    cv_mean, cv_std, _, _ = evaluate_with_tscv(
        df, grid_df, all_features, lgb_base_params,
        val_folds, nan_fill_fn=fill_fn
    )
    print(f'{strategy_name}: CV={cv_mean:.5f} ± {cv_std:.5f}')
```

| NaN戦略 | CV RMSLE |
|---------|---------|
| **NaN（デフォルト）** | **0.54520** |
| -999埋め | 0.54522 |
| 0埋め | 0.54525 |
| 中央値埋め | 0.54517 |

**「有意差がなければシンプルな方を選ぶ」**という方針のもと、以下の設定を確定しました。

```python
confirmed_settings = {
    'best_train_period': '直近15ヶ月',
    'best_train_start': '2015-12-12',
    'best_train_period_days': 456,
    'best_nan_strategy': 'NaN(デフォルト)',
    'best_rolling_config': '現行のみ',
}
```

---

### Step 3: モデル構築（4モデル）

4つの機械学習モデルを同一の特徴量・CV戦略で構築しました。全モデルで**ルールベースベースライン**（店舗×曜日の中央値）との比較を行いました。

#### 3-1. ルールベースベースラインの算出

まず、機械学習の効果を測るためのベースラインを設定します。

```python
# 各店舗×曜日の来客数中央値をそのまま予測値とする
store_dow_median = train_df.groupby(['air_store_id', 'dow'])['visitors'].median()
global_median = train_df['visitors'].median()

baseline_pred = valid_df.apply(
    lambda r: store_dow_median.get((r['air_store_id'], r['dow']), global_median),
    axis=1
)
baseline_rmsle = rmsle(valid_df['visitors'], baseline_pred)
print(f'ルールベースベースライン: RMSLE = {baseline_rmsle:.5f}')
```

```
ルールベースベースライン: RMSLE = 0.54977
```

#### 3-2. LightGBM

```python
import lightgbm as lgb

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
}

X_train, y_train = train_df[all_features], np.log1p(train_df['visitors'])
X_valid, y_valid = valid_df[all_features], np.log1p(valid_df['visitors'])

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

model_lgb = lgb.train(
    lgb_params, dtrain, num_boost_round=2000,
    valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)

pred_lgb = np.expm1(model_lgb.predict(X_valid))
print(f'LightGBM RMSLE: {rmsle(valid_df["visitors"], pred_lgb):.5f}')
```

```
LightGBM RMSLE: 0.50253
```

#### 3-3. XGBoost

```python
import xgboost as xgb

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.02,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
dvalid_xgb = xgb.DMatrix(X_valid, label=y_valid)

model_xgb = xgb.train(
    xgb_params, dtrain_xgb, num_boost_round=2000,
    evals=[(dvalid_xgb, 'valid')], early_stopping_rounds=100, verbose_eval=200,
)

pred_xgb = np.expm1(model_xgb.predict(dvalid_xgb))
print(f'XGBoost RMSLE: {rmsle(valid_df["visitors"], pred_xgb):.5f}')
```

```
XGBoost RMSLE: 0.48972
```

#### 3-4. RandomForest

```python
from sklearn.ensemble import RandomForestRegressor

# RandomForestはNaNを扱えないため、-999で埋める
X_train_rf = X_train.fillna(-999)
X_valid_rf = X_valid.fillna(-999)

model_rf = RandomForestRegressor(
    n_estimators=500, max_depth=20,
    min_samples_split=10, min_samples_leaf=5,
    max_features=0.8, max_samples=0.8,
    n_jobs=-1, random_state=42
)
model_rf.fit(X_train_rf, y_train)

pred_rf = np.expm1(model_rf.predict(X_valid_rf))
print(f'RandomForest RMSLE: {rmsle(valid_df["visitors"], pred_rf):.5f}')
```

```
RandomForest RMSLE: 0.51215
```

#### 3-5. CatBoost

```python
from catboost import CatBoostRegressor, Pool

model_cb = CatBoostRegressor(
    depth=8, learning_rate=0.02, iterations=2000,
    l2_leaf_reg=3.0, subsample=0.8, random_strength=1.0,
    min_data_in_leaf=20, eval_metric='RMSE',
    early_stopping_rounds=100, verbose=200, random_seed=42,
)

model_cb.fit(
    Pool(X_train, label=y_train),
    eval_set=Pool(X_valid, label=y_valid),
)

pred_cb = np.expm1(model_cb.predict(X_valid))
print(f'CatBoost RMSLE: {rmsle(valid_df["visitors"], pred_cb):.5f}')
```

```
CatBoost RMSLE: 0.49721
```

#### 3-6. 5-Fold CVによる汎化性能の評価

各モデルについて、Step 1で設計した5-fold時系列CVでスコアを算出しました。

```python
cv_scores = []
for i, fold in enumerate(val_folds, 1):
    val_start = pd.Timestamp(fold['val_start'])
    val_end = pd.Timestamp(fold['val_end'])

    # 学習データ: val_start より前のデータ
    fold_train = full_df[full_df['visit_date'] < val_start]
    fold_valid = full_df[
        (full_df['visit_date'] >= val_start) & (full_df['visit_date'] <= val_end)
    ]

    X_tr = fold_train[all_features]
    y_tr = np.log1p(fold_train['visitors'])
    X_va = fold_valid[all_features]

    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=np.log1p(fold_valid['visitors']), reference=dtr)

    m = lgb.train(
        lgb_params, dtr, num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred = np.expm1(m.predict(X_va))
    score = rmsle(fold_valid['visitors'], pred)
    cv_scores.append(score)
    print(f'  Fold {i}: RMSLE = {score:.5f}')

print(f'CV平均: {np.mean(cv_scores):.5f} ± {np.std(cv_scores):.5f}')
```

```
  Fold 1: 0.50557  (GW期間)
  Fold 2: 0.50840  (夏季)
  Fold 3: 0.48354  (秋季)
  Fold 4: 0.51849  (年末年始 ← 最も困難)
  Fold 5: 0.48839  (テスト直前期)
CV平均: 0.50088 ± 0.01300
```

全モデルで**Fold 4（年末年始）が最も高いRMSLE**を示し、特殊期間の予測困難さが一貫して確認されました。

#### 3-7. Optunaによるハイパーパラメータ最適化

各モデルに対してOptunaで50回（RFは30回）のCV最適化を実施しました。目的関数は**5-fold CV平均を最小化**する設計です。

```python
import optuna

def objective_lgb(trial):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbose': -1,
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
        val_start = pd.Timestamp(fold['val_start'])
        val_end = pd.Timestamp(fold['val_end'])

        fold_train = full_df[full_df['visit_date'] < val_start]
        fold_valid = full_df[
            (full_df['visit_date'] >= val_start) & (full_df['visit_date'] <= val_end)
        ]

        dtr = lgb.Dataset(fold_train[all_features], label=np.log1p(fold_train['visitors']))
        dva = lgb.Dataset(fold_valid[all_features],
                          label=np.log1p(fold_valid['visitors']), reference=dtr)

        m = lgb.train(params, dtr, num_boost_round=2000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        pred = np.expm1(m.predict(fold_valid[all_features]))
        fold_scores.append(rmsle(fold_valid['visitors'], pred))

    return np.mean(fold_scores)

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

print(f'Best CV: {study.best_value:.5f}')
print(f'Best params: {study.best_params}')
```

| モデル | デフォルト RMSLE | チューニング後 | 改善幅 |
|--------|:---:|:---:|:---:|
| LightGBM | 0.50253 | 0.50253※ | — |
| **XGBoost** | 0.48972 | **0.48917** | +0.00055 |
| RandomForest | 0.51215 | 0.50698 | +0.00517 |
| CatBoost | 0.49721 | 0.49441 | +0.00280 |

※ LightGBMはOptunaチューニングでSingle Splitが悪化したため、デフォルトパラメータを採用

---

### Step 4: モデル比較・アンサンブル

#### 4-1. 全モデルの横断比較

```python
# 03-1〜03-4の結果を読み込み
results = {}
model_files = {
    'LightGBM': '03-1_lgbm_results.pkl',
    'XGBoost': '03-2_xgb_results.pkl',
    'RandomForest': '03-3_rf_results.pkl',
    'CatBoost': '03-4_catboost_results.pkl',
}
for name, fname in model_files.items():
    with open(INTERMEDIATE_DIR / fname, 'rb') as f:
        results[name] = pickle.load(f)

# スコア一覧テーブル
summary_rows = []
for name, r in results.items():
    summary_rows.append({
        'モデル': name,
        'デフォルト RMSLE': r['score_single'],
        'チューニング後 RMSLE': r.get('tuned_score', r['score_single']),
        'CV平均': r['cv_mean'],
        '改善幅': r['score_single'] - r.get('tuned_score', r['score_single']),
    })

summary = pd.DataFrame(summary_rows).sort_values('チューニング後 RMSLE')
print(summary.to_string(index=False))
```

| モデル | デフォルト RMSLE | チューニング後 RMSLE | CV平均 | ルールベースからの改善 |
|--------|:---:|:---:|:---:|:---:|
| ルールベース | 0.54977 | — | — | — |
| LightGBM | 0.50253 | 0.50253 | 0.50088 | +0.04724 |
| **XGBoost** | 0.48972 | **0.48917** | 0.50196 | +0.06060 |
| RandomForest | 0.51215 | 0.50698 | 0.51057 | +0.04279 |
| CatBoost | 0.49721 | 0.49441 | 0.50241 | +0.05536 |

#### 4-2. 予測値の相関分析

```python
# チューニング済みモデルの予測値をDataFrameに集約
pred_df = pd.DataFrame({
    name: r.get('tuned_pred', r['valid_pred']) for name, r in results.items()
})

# 予測相関ヒートマップ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(pred_df.corr(), annot=True, fmt='.4f', cmap='YlOrRd',
            ax=axes[0], square=True, vmin=0.9, vmax=1.0)
axes[0].set_title('予測値の相関')

# 残差相関ヒートマップ
resid_df = pd.DataFrame({
    name: r.get('tuned_residuals', r['residuals']) for name, r in results.items()
})
sns.heatmap(resid_df.corr(), annot=True, fmt='.4f', cmap='YlOrRd',
            ax=axes[1], square=True, vmin=0.5, vmax=1.0)
axes[1].set_title('残差の相関')
plt.tight_layout()
plt.show()
```

モデル間の予測相関は**0.96〜0.99**と非常に高い結果でした。RandomForestはバギングベースであるため相対的に低い残差相関を示し、アンサンブルの多様性に貢献します。

#### 4-3. Ablation Study（特徴量カテゴリ別の寄与度）

特徴量カテゴリを1つずつ除外してモデルを再学習し、スコアの悪化幅で重要度を評価しました。

```python
def categorize_feature(name):
    """特徴量名からカテゴリを判定"""
    if name.startswith('rolling_') or name.startswith('ewm_'):
        return 'Rolling統計量'
    elif name.startswith('lag_'):
        return 'ラグ特徴量'
    elif name.startswith(('open_ratio_', 'closed_streak', 'days_since_')):
        return '休業パターン'
    elif name.startswith(('is_after_holiday_', 'holiday_count_', 'is_near_special')):
        return '祝日前後'
    elif name.startswith('store_'):
        return '店舗統計量'
    elif name.startswith('genre_'):
        return 'ジャンル'
    else:
        return '時間/店舗属性'

# カテゴリごとに除外して再学習
ablation_results = {}
for cat_name, cat_features in categories.items():
    remaining = [f for f in all_features if f not in cat_features]
    dtr = lgb.Dataset(train_df[remaining], label=y_train_log)
    dva = lgb.Dataset(valid_df[remaining], label=y_valid_log, reference=dtr)
    m = lgb.train(lgb_params, dtr, num_boost_round=2000,
                  valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.expm1(m.predict(valid_df[remaining]))
    score = rmsle(valid_df['visitors'], pred)
    degradation = score - base_score
    ablation_results[cat_name] = {'score': score, 'degradation': degradation}
    print(f'{cat_name} を除外: RMSLE={score:.5f} (悪化幅: {degradation:+.5f})')
```

| カテゴリ | 除外時の悪化幅 | 評価 |
|----------|:---:|------|
| Rolling統計量 | +0.01112 | **最重要** |
| 時間/店舗属性 | +0.00475 | 重要 |
| ラグ特徴量 | +0.00232 | 中程度 |
| ジャンル | -0.00043 | 寄与なし |
| 休業パターン | -0.00077 | 寄与なし |
| 祝日前後 | -0.00113 | 寄与なし |
| 店舗統計量 | -0.00917 | 除外で改善（過学習の可能性） |

#### 4-4. アンサンブル戦略の検証

まず、全モデルの組み合わせの単純平均を網羅的に評価しました。

```python
from itertools import combinations

tuned_preds = {
    name: r.get('tuned_pred', r['valid_pred']) for name, r in results.items()
}
model_list = list(tuned_preds.keys())
actual = valid_df['visitors'].values

# 全組み合わせ（2モデル〜4モデル）の単純平均
ensemble_results = []
for r_size in range(2, len(model_list) + 1):
    for combo in combinations(model_list, r_size):
        avg_pred = np.mean([tuned_preds[n] for n in combo], axis=0)
        score = rmsle(actual, avg_pred)
        ensemble_results.append({
            '組み合わせ': ' + '.join(combo),
            'RMSLE': score,
        })

ens_df = pd.DataFrame(ensemble_results).sort_values('RMSLE')
print('=== 単純平均アンサンブル（上位5） ===')
print(ens_df.head(5).to_string(index=False))
```

| 組み合わせ | RMSLE |
|-----------|:---:|
| **XGBoost + CatBoost** | **0.48739** |
| XGBoost + RF + CatBoost | 0.48824 |
| XGBoost + RF | 0.48915 |
| LightGBM + XGBoost + CatBoost | 0.48948 |
| 全4モデル | 0.48987 |

次に、Optunaで最適な重み配分を探索しました。

```python
import optuna

def objective_weight(trial):
    weights = {}
    for name in model_list:
        weights[name] = trial.suggest_float(f'w_{name}', 0.0, 1.0)
    total = sum(weights.values())
    if total == 0:
        return 999
    weighted_pred = sum(w / total * tuned_preds[n] for n, w in weights.items())
    return rmsle(actual, weighted_pred)

study_w = optuna.create_study(direction='minimize',
                              sampler=optuna.samplers.TPESampler(seed=42))
study_w.optimize(objective_weight, n_trials=200, show_progress_bar=True)

# 最適重みを正規化して表示
bp = study_w.best_params
total = sum(bp[f'w_{n}'] for n in model_list)
best_weights = {n: bp[f'w_{n}'] / total for n in model_list}

print(f'\n最適重み付きアンサンブル RMSLE: {study_w.best_value:.5f}')
for name, w in best_weights.items():
    print(f'  {name}: {w:.1%}')
```

```
最適重み付きアンサンブル RMSLE: 0.48738
  LightGBM: 0.3%
  XGBoost: 46.4%
  CatBoost: 51.2%
  RandomForest: 2.0%
```

**XGBoost（46.4%）とCatBoost（51.2%）の2モデルが支配的**であり、単純平均（0.48739）とほぼ同等のスコアとなりました。

#### 4-5. 最終スコアの可視化

```python
# 単体モデル + アンサンブルの比較棒グラフ
compare = {
    'ルールベース': 0.54977,
    'RandomForest': 0.50698,
    'LightGBM': 0.50253,
    'CatBoost': 0.49441,
    'XGBoost': 0.48917,
    '単純平均\n(XGB+Cat)': 0.48739,
    '最適重み付き': 0.48738,
}

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(compare.keys(), compare.values(),
              color=['gray', '#55A868', '#4C72B0', '#C44E52', '#DD8452', '#8172B3', '#E5AE38'])
for bar, val in zip(bars, compare.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('RMSLE')
ax.set_title('最終スコア比較')
ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
plt.show()
```

---

## 分析したデータから得られた情報

### 傾向

1. **曜日効果が支配的**: 来客数の変動の大部分は曜日パターンで説明される。店舗×曜日の中央値だけでRMSLE=0.549を達成でき、これがベースラインとなる
2. **祝日のサバイバー効果**: 祝日に営業する店舗は通常の1.3倍の来客を受ける。ただし、効果はジャンルにより大きく異なる（居酒屋: 1.48倍 vs カフェ: 1.04倍）
3. **年末年始が最も予測困難**: 全モデルでFold 4（年末年始期間）のスコアが最も悪い。通常パターンから大きく逸脱する期間は機械学習でも予測が困難
4. **特徴量重要度の偏り**: store_dow_medianとstore_dow_meanの2特徴量だけでLightGBMの重要度の**83.2%**を占める

### 分類

モデルの特性により、以下のように分類できます。

| 分類 | モデル | 特徴 |
|------|--------|------|
| 高精度・単体最強 | XGBoost | Single Split最良（0.489） |
| CV安定・汎化性能 | LightGBM | CV平均最良（0.501） |
| 多様性提供 | RandomForest | バギングによる異なる学習パラダイム |
| バランス型 | CatBoost | 単体2位・アンサンブル相性良好 |

---

## 課題

### 1. ゴールデンウィーク予測の困難さ

テスト期間の39日間のうち10.3%が祝日（GW）ですが、学習データには1回分のGWしか含まれません。サンプル不足により、GW期間の予測精度向上には限界があります。

### 2. 予約データの活用制約

予約データは全来客実績の11.1%しかカバーしておらず、HPGとAirの店舗紐づけも150店舗に限られます。予約データを特徴量として活用する効果は限定的です。

### 3. 特徴量重要度の極端な偏り

店舗統計量（特にstore_dow_median）への依存度が極めて高く、Ablation Studyでは店舗統計量を除外した方がスコアが改善する結果も得られました。過学習のリスクがあり、特徴量の再設計が必要です。

### 4. Optunaチューニングの限定的効果

4モデルともOptuna最適化による改善は0.0004〜0.005程度で、CVの標準偏差（約0.013）と比較して統計的に有意とは言えません。LightGBMではチューニング後にSingle Splitが悪化し、デフォルトパラメータを採用する判断となりました。

---

## 考察

### ルールベース vs 機械学習

店舗×曜日の中央値という極めてシンプルなルールベース予測（RMSLE=0.549）に対し、最良の機械学習アンサンブル（RMSLE=0.487）は**約11.3%の改善**を達成しました。この改善の大部分はRolling統計量とラグ特徴量によるもので、直近のトレンド変化を捉える能力が機械学習の優位性です。

### アンサンブルの効果と限界

最良単体モデル（XGBoost: 0.489）からアンサンブル（0.487）への改善は**わずか0.002**にとどまります。これはモデル間の予測相関が0.96以上と非常に高く、異なるモデルが類似の誤りを犯しているためです。さらなる改善には、異なるアプローチ（Prophet等の時系列専用モデル、外部データの活用）が必要と考えられます。

### 上位解法との比較

本分析の最良スコア（RMSLE=0.487、検証データ上）は、コンペティションの上位解法（Private LBで0.505〜0.509程度）と比較して遜色のない水準です。ただし、テストデータ上のスコアはCV値より悪化する傾向があるため、実際の提出スコアは0.50前後になると推測されます。

### 特徴量設計の重要性

モデル選択やハイパーパラメータチューニングよりも、**特徴量設計がスコアに与える影響が圧倒的に大きい**ことが確認されました。

| 改善の要因 | RMSLE改善幅 | 割合 |
|-----------|:-----------:|:---:|
| 特徴量設計（時間のみ → 全特徴量） | -0.256 | **96.8%** |
| モデル選択（LightGBM → XGBoost） | -0.006 | 2.3% |
| ハイパーパラメータ最適化 | -0.001 | 0.4% |
| アンサンブル | -0.002 | 0.8% |

---

## まとめ

本分析を通じて、以下のことがわかりました。

1. **飲食店の来客数は、曜日パターンと直近トレンドで大部分が説明できる** — 店舗×曜日の中央値だけでRMSLE=0.549、Rolling/ラグ特徴量の追加でさらに0.05改善
2. **特徴量設計が最も重要** — 改善の97%は特徴量に起因し、モデル選択やチューニングの寄与は限定的
3. **XGBoostとCatBoostの2モデルアンサンブルが最適** — 最終RMSLE=0.487（ルールベースから11.3%改善）
4. **祝日・特殊期間の予測は依然として困難** — GWや年末年始はサンプル不足と挙動の特殊性から、全モデルで精度が低下

### 今後の改善方針

- **外部データの活用**: 天気データ、イベント情報の統合
- **時系列専用モデル**: Prophet、N-BEATSなどの導入
- **店舗クラスタリング**: 類似店舗のグループ化による情報共有
- **ジャンル別モデル**: 居酒屋とカフェなど、挙動の異なるジャンルで別モデルを構築

---

## 参考にした資料・文献

- [Kaggle: Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/competitions/recruit-restaurant-visitor-forecasting) — コンペティションページ
- [8th Place Solution (MaxHalford)](https://github.com/MaxHalford/kaggle-recruit-restaurant) — LightGBM + 特徴量エンジニアリング（Public 0.468 / Private 0.509）
- [11th Place Solution (XIUQI1994)](https://github.com/XIUQI1994/Kaggle_Recruit-Restaurant-Visitor-Forecasting_) — Prophet + 天気データ統合
- [Kaggle Days 3rd Place Solution](https://github.com/dkivaranovic/kaggledays-recruit) — LightGBM + Keras アンサンブル（Final 0.505）
- 農林水産省「食品ロス量の推計値（令和4年度）」
