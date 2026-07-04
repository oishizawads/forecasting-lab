"""Forecasting Lab の Streamlit エントリポイント。

UI は薄く保ち、予測・評価ロジックは forecasting_lab パッケージ(src/) に委譲する。
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# src/ 配下のパッケージをインポート可能にする（未インストール環境でも動作する保险）
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting_lab import data as ds
from forecasting_lab.metrics import evaluate
from forecasting_lab.models import LinearTrendModel, MovingAverageModel, NaiveModel
from forecasting_lab.brand import (
    apply_brand,
    footer_backlink,
    hero,
    plotly_template,
    section,
    sidebar_header,
    PALETTE,
    show_table,
)

st.set_page_config(page_title="Forecasting Lab", page_icon="📈", layout="centered")
apply_brand(st)

# モデル表示名 -> 生成器（窓幅 w を受け取る）
MODEL_BUILDERS = {
    "Naive": lambda w: NaiveModel(),
    "Moving Average": lambda w: MovingAverageModel(window=w),
    "Linear Trend": lambda w: LinearTrendModel(),
}
MODEL_COLORS = {
    "Naive":         (PALETTE[0], "rgba(15,118,110,0.15)"),
    "Moving Average":(PALETTE[1], "rgba(37,99,235,0.15)"),
    "Linear Trend":  (PALETTE[2], "rgba(217,119,6,0.15)"),
}
Z_BY_LEVEL = {"80%": 1.28, "90%": 1.645, "95%": 1.96}
DATASET_OPTIONS = ds.DATASET_NAMES + ["Upload CSV"]


@st.cache_data(show_spinner=False)
def get_dataset(name: str, n: int, seed: int, file_bytes: bytes | None) -> pd.DataFrame | None:
    """データセットを取得する。CSV の場合はバイト列から復元する。"""
    if name == "Upload CSV":
        if not file_bytes:
            return None
        df = pd.read_csv(io.BytesIO(file_bytes))
        if df.shape[1] < 2:
            return None
        df = df[[df.columns[0], df.columns[1]]].copy()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    return ds.load_dataset(name, n=n, seed=seed)


@st.cache_data(show_spinner=False)
def run_forecast(
    values: tuple[float, ...],
    n_train: int,
    ma_window: int,
    model_names: tuple[str, ...],
    z: float,
) -> dict:
    """選択されたモデルで予測を実行し、予測値・区間・評価指標・残差を返す。"""
    arr = np.asarray(values, dtype=float)
    results: dict[str, dict] = {}
    for mname in model_names:
        builder = MODEL_BUILDERS[mname]
        try:
            model = builder(ma_window).fit(arr[:n_train])
        except ValueError as exc:
            results[mname] = {"error": str(exc)}
            continue
        horizon = arr.size - n_train
        mean, lower, upper = model.predict_interval(horizon, z=z)
        y_test = arr[n_train:]
        results[mname] = {
            "in_sample": model.predict_in_sample(),
            "forecast": mean,
            "lower": lower,
            "upper": upper,
            "metrics": evaluate(y_test, mean),
            "residuals": y_test - mean,
        }
    return results


def _format_metric(val: float) -> str:
    if not np.isfinite(val):
        return "—"
    return f"{val:.3f}"


# --------------------------------------------------------------------- Hero
hero(
    st,
    "TIME SERIES",
    "Forecasting Lab",
    "単純な時系列予測モデルを比較し、予測精度と予測区間を可視化します。",
    chips=["Python", "Plotly", "Statsmodels", "stlite"],
)

# --------------------------------------------------------------------- Sidebar
with st.sidebar:
    sidebar_header(st, "Forecasting Lab")
    data_source = st.selectbox("データセット", DATASET_OPTIONS)
    file_bytes: bytes | None = None
    if data_source == "Upload CSV":
        up = st.file_uploader("CSV をアップロード（先頭2列: date, value）", type=["csv"])
        if up is not None:
            file_bytes = up.getvalue()
    is_synth = data_source != "Upload CSV"
    n_points = st.slider("データ数", 36, 240, 120, step=12, disabled=not is_synth)
    seed = st.number_input("乱数シード", 0, 100, 0, disabled=not is_synth)
    test_size = st.slider("評価期間の割合", 0.1, 0.5, 0.25, 0.05)
    ma_window = st.number_input("Moving Average の窓幅", 1, 24, 3)
    conf = st.selectbox("予測区間の信頼水準", list(Z_BY_LEVEL.keys()), index=2)
    models = st.multiselect(
        "予測モデル", list(MODEL_BUILDERS.keys()), default=list(MODEL_BUILDERS.keys())
    )

z = Z_BY_LEVEL[conf]

df = get_dataset(data_source, int(n_points), int(seed), file_bytes)
if df is None or df.empty:
    st.warning("データがありません。CSV をアップロードするか、データセットを選択してください。")
    st.stop()

values = df["value"].to_numpy(dtype=float)
n = len(values)
valid_count = int(np.isfinite(values).sum())

if valid_count < 2:
    st.error("有効な観測値が2件未満です。別のデータセットまたはCSVを使用してください。")
    st.stop()

n_train = max(1, int(n * (1 - test_size)))
if n_train >= n:
    st.error("評価期間が確保できません。データ数を増やすか、評価期間の割合を下げてください。")
    st.stop()

if valid_count < n:
    st.info(f"欠損値を含むデータです（有効 {valid_count}/{n} 件）。予測時は前方補完で扱います。")

if not models:
    st.info("左サイドバーから予測モデルを1つ以上選択してください。")
    st.stop()

dates = df["date"].to_numpy()
train_dates, test_dates = dates[:n_train], dates[n_train:]
results = run_forecast(
    tuple(values.tolist()), n_train, int(ma_window), tuple(models), float(z)
)

# --- データプレビュー
with st.expander("データプレビュー", expanded=False):
    show_table(st, df.assign(value=df["value"].round(3)))

# --- 予測グラフ
section(st, "FORECAST", "予測グラフ（学習期間 / 評価期間 / 予測区間）")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=train_dates, y=values[:n_train], name="学習（実測）",
        mode="lines+markers", line=dict(color="#9aa0a6", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=test_dates, y=values[n_train:], name="評価（実測）",
        mode="lines+markers", line=dict(color="#0d1526", width=2),
    )
)
for mname, res in results.items():
    if "error" in res:
        continue
    line_color, fill_color = MODEL_COLORS[mname]
    fig.add_trace(
        go.Scatter(
            x=test_dates, y=res["upper"], name=f"{mname} 上限",
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates, y=res["lower"], name=f"{mname} 下限",
            mode="lines", line=dict(width=0), fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates, y=res["forecast"], name=f"{mname} 予測",
            mode="lines+markers", line=dict(color=line_color, width=2.5),
        )
    )
fig.update_layout(**plotly_template())
fig.update_layout(
    xaxis_title="日付",
    yaxis_title="値",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# --- 評価指標
section(st, "METRICS", "評価指標（評価期間）")
rows = []
for mname, res in results.items():
    if "error" in res:
        rows.append({"モデル": mname, "MAE": "—", "RMSE": "—", "備考": res["error"]})
    else:
        m = res["metrics"]
        rows.append(
            {
                "モデル": mname,
                "MAE": _format_metric(m["MAE"]),
                "RMSE": _format_metric(m["RMSE"]),
                "備考": "",
            }
        )
metrics_df = pd.DataFrame(rows)
show_table(st, metrics_df, float_fmt="{:.3f}")

st.download_button(
    "評価指標をCSVでダウンロード",
    metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="forecasting_metrics.csv",
    mime="text/csv",
)

# --- 残差プロット
section(st, "RESIDUALS", "残差プロット（評価期間）")
fig2 = go.Figure()
for mname, res in results.items():
    if "error" in res:
        continue
    fig2.add_trace(
        go.Scatter(
            x=test_dates, y=res["residuals"], name=f"{mname} 残差",
            mode="lines+markers",
        )
    )
fig2.add_hline(y=0, line_dash="dash", line_color="#9aa0a6")
fig2.update_layout(**plotly_template())
fig2.update_layout(
    xaxis_title="日付",
    yaxis_title="残差",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=320,
)
st.plotly_chart(fig2, use_container_width=True)

footer_backlink(st, repo="forecasting-lab")
