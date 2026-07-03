# Forecasting Lab

## 目的

単純な時系列予測モデルを並べて比較し、**予測精度（MAE/RMSE）**と**予測区間**を視覚的に確認するための小型アプリ。ポートフォリオ向けの MVP であり、複雑な状態管理や本番運用機能は持たない。

## 主要機能

- データセットの選択（合成データ4種 + ユーザー任意のCSVアップロード）
- 予測モデルの選択・同時比較
  - Naive Forecast（直前値を引き伸ばす）
  - Moving Average（直近 N 件の平均）
  - Linear Trend（OLS で線形トレンドを外挿）
- 学習期間と評価期間を分けて表示した予測グラフ（予測区間バンド付き）
- 評価指標テーブル（MAE / RMSE）と CSV ダウンロード
- 残差プロット（評価期間）
- 欠損値を含むデータでも前方補完で動作し、落ちない

## 使用技術

- Python 3.11+
- Streamlit（UI）
- pandas / NumPy（データ処理）
- scikit-learn / statsmodels（依存に含むが、本MVPの予測ロジックはNumPyで実装）
- Plotly（インタラクティブなグラフ、モバイル幅にも追従）
- pytest（テスト）

## データの出所

本アプリは**合成データ**を既定で使用する（`src/forecasting_lab/data.py` の乱数生成器）。外部の公開データセットには依存しない。ユーザーが独自のCSV（先頭2列を `date, value` とする）をアップロードすることも可能。外部データを取り込む場合は、出典とライセンスを別途明記すること。

## ローカル実行手順

```bash
# 1) 仮想環境の作成（uv を使う場合）
uv venv --python 3.11
source .venv/bin/activate

# 2) 依存関係のインストール
pip install -r requirements.txt
# または: uv pip install -r requirements.txt

# 3) アプリの起動
streamlit run app/streamlit_app.py
```

ブラウザで `http://localhost:8501` が開き、左サイドバーからデータセット・モデル・評価期間などを設定できる。

### テスト

```bash
pytest
```

`src/forecasting_lab/` の予測・評価関数に対する最低限のテストが実行される。

## スクリーンショット

スクリーンショットは `assets/` に配置する（現在は空）。

## 制限事項

- 予測区間は学習データの in-sample 残差標準偏差を用いた対称バンドの近似であり、厳密な統計区間ではない。
- Naive / Moving Average はフラット予測、Linear Trend は線形外挿のみで、季節性や自己回帰構造は扱わない。
- CSV の列解釈は先頭2列を固定で `date, value` とみなす。
- 認証・DB・課金・本番運用機能は範囲外。

## 免責

本アプリは**単純モデルの比較**を目的としており、**実運用の予測精度を保証するものではない**。
