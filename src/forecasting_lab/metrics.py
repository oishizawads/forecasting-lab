"""予測精度評価指標（MAE / RMSE）。欠損値を含むデータでも落ちない。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_array(x) -> np.ndarray:
    """Series/配列を 1次元 float 配列に正規化する。"""
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float).ravel()


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """2つの配列の長期を検証し、float 配列として返す。"""
    yt = _to_array(y_true)
    yp = _to_array(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"length mismatch: y_true={yt.shape[0]} y_pred={yp.shape[0]}")
    return yt, yp


def _valid_mask(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """両側が有限（NaN/inf でない）のペアを示すマスク。"""
    return np.isfinite(yt) & np.isfinite(yp)


def mae(y_true, y_pred) -> float:
    """平均絶対誤差。有効ペアが無い場合は NaN を返す。"""
    yt, yp = _validate_pair(y_true, y_pred)
    mask = _valid_mask(yt, yp)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask])))


def rmse(y_true, y_pred) -> float:
    """二乗平均平方根誤差。有効ペアが無い場合は NaN を返す。"""
    yt, yp = _validate_pair(y_true, y_pred)
    mask = _valid_mask(yt, yp)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((yt[mask] - yp[mask]) ** 2)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """MAE と RMSE をまとめて返す。"""
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred)}
