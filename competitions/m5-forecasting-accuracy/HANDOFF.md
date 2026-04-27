# M5 Forecasting Accuracy - 引き継ぎサマリー

**作成日**: 2026-04-27
**最終マージコミット**: `7249ef1` (PR #3, mainマージ済み)

---

## 1. これまでの作業

### Phase 1: 2時間ベースライン (完了)
- LightGBM Tweedie 単一fold モデルを作成
- Drive/Colab経由で実行（ローカルメモリ不足回避）
- **private LB: 0.64873** で初提出
- ファイル: `notebooks/m5_colab_pipeline.ipynb` (旧版)

### Phase 2: Plan C フルパッケージ着手 (進行中)
ユーザーが選択した路線:
- 中央値 + LightGBM + XGBoost + CatBoost + 店舗別RandomForest
- 各モデル 5-fold CV + Optuna 30 trials
- セグメント別ウェイト最適化アンサンブル
- 想定スコア: **0.55〜0.58（銀メダル圏）**

---

## 2. 現在のブロッカー

### 01_features.ipynb の最終セル

Drive上の `01_features.ipynb` を Colab で実行中、セル20まで成功（`(20428300, 57)` 5531MB）したが**最終セル(parquet保存)が `.copy()` でOOM**。

**修正版コード**（最終セルに貼り付け実行で解決）:

```python
import pyarrow as pa
import pyarrow.parquet as pq

keep = ['id','d_num','sales','store_id','item_id','dept_id','cat_id','state_id'] + feature_cols
df = df[keep]
gc.collect()
print(f'after column drop mem={df.memory_usage(deep=True).sum()/1024**2:.0f}MB')

out_path = CACHE / 'features_all.parquet'
chunk_size = 2_000_000
writer = None
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
    writer.write_table(table)
    del chunk, table
    gc.collect()
    print(f'  {min(i+chunk_size, len(df))}/{len(df)} rows written')
writer.close()

size_mb = out_path.stat().st_size / 1024**2
print(f'\nparquet size: {size_mb:.0f} MB')

with open(CACHE/'val_folds.pkl','wb') as f: pickle.dump(val_folds, f)
with open(CACHE/'feature_cols.pkl','wb') as f: pickle.dump(feature_cols, f)
print(f'\n全工程: {(time.time()-T0)/60:.1f}分')
print('=== 完了 ===')
```

---

## 3. 残タスク

| # | タスク | 想定時間 | 状態 |
|---|---|---|---|
| 1 | Drive上 `01_features.ipynb` の最終セル修正版を実行 | 〜10分 | 🟡 ブロック中 |
| 2 | `02_gbdt_models.ipynb` 作成 (中央値+LGBM+XGB+CatBoost, 各Optuna 30 trials) | 〜10時間 (Colab) | ⬜ |
| 3 | `03_rf_per_store.ipynb` 作成 (店舗別RF 10モデル) | 〜2-3時間 (Colab) | ⬜ |
| 4 | `04_ensemble_submit.ipynb` 作成 (ウェイト最適化+提出) | 〜30分 (Colab) | ⬜ |

---

## 4. リソース・環境情報

### Google Drive
- ルート: `MyDrive/Colab Data/m5-forecasting-accuracy/`
- フォルダID: `166iWsDcSIk9SDpblhCuDn1yWS0M_tCeW`
- 入力CSV配置済み: calendar / sell_prices / sales_train_evaluation / sample_submission
- ノートブック配置:
  - `m5_colab_pipeline.ipynb` (旧2時間版, ID: `1c-MU9MV0CAWBqDmgU79q0UaAM0nbdWmk`)
  - `01_features.ipynb` (Plan C 特徴量パイプライン, ID: `1UcRRWYQCnn_EeNepuLgDNVOxbyfSMatV`)

### Colab Runtime
- **Pro 契約済み（ハイメモリ 51GB 利用可）**
- ハードウェア: CPU で十分（GPU不要）

### Kaggle CLI
- インストール済み (v2.0.0)
- `kaggle.json` 配置済み (`~/.kaggle/kaggle.json`)
- 提出コマンド: `kaggle competitions submit -c m5-forecasting-accuracy -f output/submission.csv -m "..."`

### MCP
- Google Drive MCP 認証済み（claude.aiコネクタ）
- 別PCで `/mcp` 再認証が必要

---

## 5. 新PCでの再開手順

### A. リポジトリ取得
```bash
git clone https://github.com/Sekine53629/kaggle.git
cd kaggle
git checkout main
git pull origin main  # 最新コミット 7249ef1 まで取得済みになる
```

### B. Python環境
```bash
pip install -r requirements.txt
# 主要: pandas, numpy, lightgbm, xgboost, catboost, scikit-learn, optuna, pyarrow, kaggle
```

### C. Kaggle CLI
- `kaggle.json` を `~/.kaggle/` (Win: `C:\Users\<user>\.kaggle\`) に配置
- `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### D. M5データダウンロード（必要なら）
```bash
cd competitions/m5-forecasting-accuracy
mkdir -p input
kaggle competitions download -c m5-forecasting-accuracy -p input/
cd input && unzip m5-forecasting-accuracy.zip && rm *.zip && cd ..
```
※ Drive上のCSVは既に正しく配置されているので、Colabのみで作業継続するならこれは不要。

### E. Claude Code 再開
1. プロジェクトディレクトリで Claude Code 起動
2. 自動で memory/ から `m5_project_state.md` を読み込み、状態が復元される
3. `/mcp` で claude.ai Google Drive を再認証
4. 「M5の続き、Plan C 02 から」と指示

---

## 6. 重要設計判断（再現用）

### 特徴量（48個）
- カレンダー: wday, month, year, day_of_month, week_of_year, quarter, is_weekend, has_event, event_name_code, event_type_code, snap (11個)
- 価格: sell_price, price_max/min/mean/std/discount/norm/z (8個)
- ラグ: lag_28, 29, 30, 35, 42, 49, 56, 63, 70 (9個)
- ローリング (shift 28後): rmean_7/14/28/60/180, rstd_7/14/28, rmedian_28, rmax_28 (10個)
- グループ集約: store_dow_mean, item_dow_mean, dept_dow_mean, si_mean, si_std (5個)
- カテゴリエンコーディング: store_id_enc, item_id_enc, dept_id_enc, cat_id_enc, state_id_enc (5個)

### CV戦略（リーク回避）
- 5-fold 時系列CV
- fold 1: d_1886-d_1913 (直近28日, Public LB近似)
- fold 2: d_1858-d_1885
- fold 3: d_1830-d_1857
- fold 4: d_1802-d_1829
- fold 5: d_1774-d_1801

### 学習データ範囲
- d_1300 以降（前回 d_1500 から +200日拡大）

### 予測戦略
- 予測期間28日 とラグ28日が一致 → **再帰予測不要**、直接予測可能
- evaluation部分（d_1942-d_1969）はモデル予測
- validation部分（d_1914-d_1941）は実測値で埋める

---

## 7. スコア記録

| 提出 | モデル | private LB | 備考 |
|---|---|---|---|
| #1 | LightGBM Tweedie 単一fold (d>=1500) | **0.64873** | 2時間ベースライン |
| #2〜 | （Plan C 完成後） | TBD | 目標 0.55〜0.58 |
