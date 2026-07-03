"""データセット読み込み（合成データ生成 + CSV アップロード）。

本モジュールが扱うデータはすべて合成データ（またはユーザー任意のCSV）であり、
外部の公開データセットには依存しない。そのため出典/ライセンスの記載は不要。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASET_NAMES: list[str] = [
    "synthetic_trend_seasonal",
    "synthetic_trend_only",
    "synthetic_with_missing",
    "synthetic_noisy_level",
]


def _dates(n: int, start: str = "2018-01-01", freq: str = "MS") -> pd.DatetimeIndex:
    """n 期間分の月次タイムスタンプを返す。"""
    return pd.date_range(start=start, periods=n, freq=freq)


def _frame(values: np.ndarray, start: str = "2018-01-01", freq: str = "MS") -> pd.DataFrame:
    """1次元配列を date/value 列の DataFrame に整形する。"""
    n = int(len(values))
    return pd.DataFrame({"date": _dates(n, start, freq), "value": values.astype(float)})


def synthetic_trend_seasonal(
    n: int = 120, seed: int = 0, trend: float = 0.05, season_amp: float = 3.0, noise: float = 1.0
) -> pd.DataFrame:
    """トレンド + 季節性(周期12) + ノイズの合成月次系列。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = season_amp * np.sin(2.0 * np.pi * t / 12.0)
    series = 10.0 + trend * t + seasonal + noise * rng.standard_normal(n)
    return _frame(series)


def synthetic_trend_only(
    n: int = 120, seed: int = 0, trend: float = 0.1, noise: float = 0.5
) -> pd.DataFrame:
    """トレンドのみの合成月次系列。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    series = 10.0 + trend * t + noise * rng.standard_normal(n)
    return _frame(series)


def synthetic_with_missing(
    n: int = 120, seed: int = 0, missing_frac: float = 0.1
) -> pd.DataFrame:
    """欠損値を意図的に混入した合成月次系列（堅牢性確認用）。"""
    df = synthetic_trend_seasonal(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    mask = rng.random(n) < missing_frac
    df.loc[mask, "value"] = np.nan
    return df


def synthetic_noisy_level(n: int = 120, seed: int = 0, noise: float = 2.0) -> pd.DataFrame:
    """トレンドなしの水平+ノイズ系列。"""
    rng = np.random.default_rng(seed)
    series = 10.0 + noise * rng.standard_normal(n)
    return _frame(series)


_GENERATORS = {
    "synthetic_trend_seasonal": synthetic_trend_seasonal,
    "synthetic_trend_only": synthetic_trend_only,
    "synthetic_with_missing": synthetic_with_missing,
    "synthetic_noisy_level": synthetic_noisy_level,
}


def load_dataset(name: str, n: int = 120, seed: int = 0) -> pd.DataFrame:
    """名前で合成データセットを生成して返す。未知の名前は ValueError。"""
    if name not in _GENERATORS:
        raise ValueError(f"unknown dataset: {name}")
    if n < 2:
        raise ValueError("n must be >= 2")
    return _GENERATORS[name](n=n, seed=seed)


def load_csv(path: str | Path) -> pd.DataFrame:
    """ユーザー任意のCSVを読み込む。先頭2列を (date, value) として扱う。

    日付は可能な限り datetime に変換し、値は数値化する。
    """
    p = Path(path)
    df = pd.read_csv(p)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns: date, value")
    date_col, val_col = df.columns[0], df.columns[1]
    df = df[[date_col, val_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df
