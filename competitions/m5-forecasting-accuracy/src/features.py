"""M5 Forecasting - 共通特徴量生成モジュール

大容量データ処理戦略:
- 全体meltは1回だけ実行し、parquetにキャッシュ
- reduce_mem_usageでメモリ削減
- ラグ/ローリング特徴量は28日以上のラグのみ（予測期間28日のため）
"""

import gc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# パス設定（notebooks/説明用資料/ から呼ばれる想定）
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'input'
INTERMEDIATE_DIR = BASE_DIR / 'notebooks' / '説明用資料' / 'intermediate'


# ============================================================
# 1. データ読み込みとmelt
# ============================================================

def load_raw_data():
    """生データを読み込む（メモリ最適化済み）"""
    # Calendar
    calendar = pd.read_csv(INPUT_DIR / 'calendar.csv', parse_dates=['date'])
    calendar['d_num'] = calendar['d'].str[2:].astype(int)

    # Sales（dtype指定で451MB→120MB）
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [f'd_{i}' for i in range(1, 1942)]
    dtype_dict = {col: 'int16' for col in d_cols}
    dtype_dict.update({col: 'category' for col in id_cols})
    sales = pd.read_csv(INPUT_DIR / 'sales_train_evaluation.csv', dtype=dtype_dict)

    # Sell prices
    sell_prices = pd.read_csv(INPUT_DIR / 'sell_prices.csv')

    return calendar, sales, sell_prices


def melt_sales(sales, calendar):
    """salesをワイド→ロング形式に変換（~59M行）

    キャッシュファイルが存在すればそこから読み込む。
    """
    cache_path = INTERMEDIATE_DIR / 'sales_long.parquet'
    if cache_path.exists():
        print(f'キャッシュから読み込み: {cache_path}')
        return pd.read_parquet(cache_path)

    print('salesをmelt中... (初回のみ、数分かかります)')
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = [f'd_{i}' for i in range(1, 1942)]

    # melt
    df = sales.melt(id_vars=id_cols, value_vars=d_cols,
                    var_name='d', value_name='sales')

    # d番号を抽出してcalendarと結合
    df['d_num'] = df['d'].str[2:].astype(int)
    df = df.merge(
        calendar[['d_num', 'date', 'wm_yr_wk', 'wday', 'month', 'year',
                  'weekday', 'event_name_1', 'event_type_1',
                  'snap_CA', 'snap_TX', 'snap_WI']],
        on='d_num', how='left'
    )
    df.drop(columns=['d'], inplace=True)

    # メモリ最適化
    df['sales'] = df['sales'].astype('int16')
    df['d_num'] = df['d_num'].astype('int16')
    df['wday'] = df['wday'].astype('int8')
    df['month'] = df['month'].astype('int8')
    df['year'] = df['year'].astype('int16')
    df['snap_CA'] = df['snap_CA'].astype('int8')
    df['snap_TX'] = df['snap_TX'].astype('int8')
    df['snap_WI'] = df['snap_WI'].astype('int8')
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'weekday', 'event_name_1', 'event_type_1']:
        df[col] = df[col].astype('category')

    # キャッシュ保存
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f'キャッシュ保存完了: {cache_path} ({cache_path.stat().st_size / 1024**2:.0f} MB)')

    return df


# ============================================================
# 2. 特徴量生成
# ============================================================

def add_calendar_features(df):
    """カレンダー特徴量を追加"""
    df['day_of_month'] = df['date'].dt.day.astype('int8')
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype('int8')
    df['is_weekend'] = (df['wday'].isin([1, 2])).astype('int8')  # 1=Sat, 2=Sun
    df['has_event'] = df['event_name_1'].notna().astype('int8')

    # SNAP: 州ごとの適用フラグを1列にまとめる
    df['snap'] = 0
    for state in ['CA', 'TX', 'WI']:
        mask = df['state_id'] == state
        df.loc[mask, 'snap'] = df.loc[mask, f'snap_{state}']
    df['snap'] = df['snap'].astype('int8')

    return df


def add_price_features(df, sell_prices, calendar):
    """価格特徴量を追加"""
    # wm_yr_wk結合用のマッピング
    week_map = calendar[['d_num', 'wm_yr_wk']].drop_duplicates()

    # df にwm_yr_wkがなければ追加
    if 'wm_yr_wk' not in df.columns:
        df = df.merge(week_map, on='d_num', how='left')

    # 価格結合
    df = df.merge(sell_prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
                  on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # 価格派生特徴量（商品×店舗単位）
    price_stats = sell_prices.groupby(['store_id', 'item_id'])['sell_price'].agg(
        price_max='max', price_min='min', price_std='std', price_mean='mean'
    ).reset_index()
    df = df.merge(price_stats, on=['store_id', 'item_id'], how='left')

    # 割引率（最大価格からの値下げ率）
    df['price_discount'] = (df['price_max'] - df['sell_price']) / df['price_max']
    # 平均からの乖離率
    df['price_deviation'] = (df['sell_price'] - df['price_mean']) / df['price_mean']

    # メモリ最適化
    for col in ['sell_price', 'price_max', 'price_min', 'price_std', 'price_mean',
                'price_discount', 'price_deviation']:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    return df


def add_lag_features(df, lags=None):
    """ラグ特徴量を追加

    注意: 予測期間が28日のため、lag < 28 は予測時に使えない。
    lag_28, lag_35, ... のみ使用。
    """
    if lags is None:
        lags = [28, 35, 42, 49, 56]

    print(f'ラグ特徴量を生成中: {lags}')
    df = df.sort_values(['id', 'd_num']).reset_index(drop=True)

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag).astype('float32')

    return df


def add_rolling_features(df, windows=None, lag_base=28):
    """ローリング特徴量を追加

    lag_baseで指定した日数分シフトしてからrollingを計算。
    これにより予測時にも使える特徴量になる。
    """
    if windows is None:
        windows = [7, 14, 28, 60]

    print(f'ローリング特徴量を生成中: windows={windows}, lag_base={lag_base}')
    df = df.sort_values(['id', 'd_num']).reset_index(drop=True)

    # lag_base日分シフトした販売数
    shifted = df.groupby('id')['sales'].shift(lag_base)

    for w in windows:
        df[f'rolling_mean_{w}'] = (
            shifted.groupby(df['id']).transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            ).astype('float32')
        )
        if w <= 28:
            df[f'rolling_std_{w}'] = (
                shifted.groupby(df['id']).transform(
                    lambda x: x.rolling(w, min_periods=1).std()
                ).astype('float32')
            )

    return df


def add_encoding_features(df):
    """カテゴリエンコーディング特徴量

    Lesson 7対策: 学習データのマッピングを構築して適用。
    ここでは全データで一括factorizeするため問題なし。
    train/test分割後に独立factorizeしないこと。
    """
    for col in ['store_id', 'item_id', 'dept_id', 'cat_id', 'state_id']:
        df[f'{col}_enc'] = df[col].astype(str).factorize()[0].astype('int16')

    return df


# ============================================================
# 3. CV戦略（Lesson 2: 全ノートブックで統一）
# ============================================================

def get_val_folds():
    """時系列CVのfold定義

    テスト期間: 28日（d_1914~d_1941）
    各foldも28日のvalidation window。
    """
    val_folds = [
        {'val_start': '2016-03-28', 'val_end': '2016-04-24', 'fold': 1},  # d_1886~d_1913
        {'val_start': '2016-02-29', 'val_end': '2016-03-27', 'fold': 2},  # d_1858~d_1885
        {'val_start': '2016-01-31', 'val_end': '2016-02-28', 'fold': 3},  # d_1829~d_1857
        {'val_start': '2016-01-03', 'val_end': '2016-01-30', 'fold': 4},  # d_1801~d_1828
        {'val_start': '2015-12-06', 'val_end': '2016-01-02', 'fold': 5},  # d_1773~d_1800 (年末含む)
    ]
    return val_folds


# ============================================================
# 4. 特徴量リスト定義
# ============================================================

def get_feature_cols():
    """モデルに入力する特徴量列のリスト"""
    features = [
        # カレンダー
        'wday', 'month', 'year', 'day_of_month', 'week_of_year',
        'is_weekend', 'has_event', 'snap',
        # 価格
        'sell_price', 'price_max', 'price_min', 'price_std', 'price_mean',
        'price_discount', 'price_deviation',
        # ラグ
        'lag_28', 'lag_35', 'lag_42', 'lag_49', 'lag_56',
        # ローリング
        'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_28', 'rolling_mean_60',
        'rolling_std_7', 'rolling_std_14', 'rolling_std_28',
        # エンコーディング
        'store_id_enc', 'item_id_enc', 'dept_id_enc', 'cat_id_enc', 'state_id_enc',
    ]
    return features


# ============================================================
# 5. パイプライン: 一括実行
# ============================================================

def build_features(use_cache=True):
    """特徴量生成パイプラインを一括実行

    Returns:
        df: 全特徴量付きDataFrame
        feature_cols: 特徴量列のリスト
        val_folds: CV fold定義
    """
    cache_path = INTERMEDIATE_DIR / 'features_all.parquet'
    feature_cols = get_feature_cols()
    val_folds = get_val_folds()

    if use_cache and cache_path.exists():
        print(f'特徴量キャッシュから読み込み: {cache_path}')
        df = pd.read_parquet(cache_path)
        return df, feature_cols, val_folds

    print('=== 特徴量生成パイプライン開始 ===')

    # Step 1: 生データ読み込み
    print('\n[1/6] 生データ読み込み...')
    calendar, sales, sell_prices = load_raw_data()

    # Step 2: melt
    print('\n[2/6] salesをmelt...')
    df = melt_sales(sales, calendar)
    del sales
    gc.collect()

    # Step 3: カレンダー特徴量
    print('\n[3/6] カレンダー特徴量...')
    df = add_calendar_features(df)

    # Step 4: 価格特徴量
    print('\n[4/6] 価格特徴量...')
    df = add_price_features(df, sell_prices, calendar)
    del sell_prices
    gc.collect()

    # Step 5: ラグ & ローリング特徴量
    print('\n[5/6] ラグ & ローリング特徴量...')
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Step 6: エンコーディング特徴量
    print('\n[6/6] エンコーディング特徴量...')
    df = add_encoding_features(df)

    # キャッシュ保存
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    size_mb = cache_path.stat().st_size / 1024**2
    print(f'\n特徴量キャッシュ保存完了: {cache_path} ({size_mb:.0f} MB)')
    print(f'DataFrame shape: {df.shape}')
    print(f'メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB')

    return df, feature_cols, val_folds


def prepare_train_data(df, feature_cols, train_start_d=None):
    """学習用データを準備

    Args:
        df: 全特徴量付きDataFrame
        feature_cols: 特徴量列リスト
        train_start_d: 学習開始日のd_num（Noneなら全期間）

    Returns:
        df_train: ラグ特徴量が有効な範囲のみ（NaN行を除外）
    """
    # 学習データ: d_1 ~ d_1913 (evaluation期間は除外)
    df_train = df[df['d_num'] <= 1913].copy()

    # 学習開始日でフィルタ
    if train_start_d is not None:
        df_train = df_train[df_train['d_num'] >= train_start_d]

    # ラグ特徴量のNaN行を除外（初期の28日+α）
    df_train = df_train.dropna(subset=['lag_28'])

    print(f'学習データ: {df_train.shape}')
    print(f'期間: d_{df_train["d_num"].min()} ~ d_{df_train["d_num"].max()}')

    return df_train
