"""予測モデル: Naive / Moving Average / Linear Trend。

すべてのモデルは欠損値を含む入力でも例外を投げずに動作する（全欠損時のみ ValueError）。
予測区間は学習データの in-sample 残差標準偏差を用いた対称バンドで近似する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _to_array(x) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.asarray(x, dtype=float).ravel()


def _has_valid(y: np.ndarray) -> bool:
    return bool(np.isfinite(y).any())


def _filled(y: np.ndarray) -> np.ndarray:
    """NaN を前方補完した配列を返す（先頭の連続 NaN はそのまま）。"""
    return pd.Series(y).ffill().to_numpy(dtype=float)


def _resid_std(observed: np.ndarray, fitted: np.ndarray) -> float:
    """両者が有限の箇所の残差標準偏差。2点未満なら 0。"""
    mask = np.isfinite(observed) & np.isfinite(fitted)
    if mask.sum() < 2:
        return 0.0
    return float(np.std(observed[mask] - fitted[mask], ddof=0))


class ForecastModel(ABC):
    """予測モデルの共通インターフェース。"""

    name: str = "base"

    @abstractmethod
    def fit(self, y) -> "ForecastModel":
        """学習系列を与えてモデルを適合させる。"""

    @abstractmethod
    def predict_in_sample(self) -> np.ndarray:
        """学習期間に対する当てはめ値（1期前シフトベース）を返す。"""

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """horizon 期間分の将来予測を返す。"""

    def residual_std(self) -> float:
        """in-sample 残差の標準偏差（予測区間用）。"""
        raise NotImplementedError

    def predict_interval(self, horizon: int, z: float = 1.96):
        """予測値と z*residual_std の対称区間を返す。

        Returns:
            (mean, lower, upper) -- いずれも長さ horizon の配列。
        """
        horizon = max(0, int(horizon))
        mean = self.predict(horizon)
        sd = self.residual_std()
        if not np.isfinite(sd) or sd <= 0:
            sd = 0.0
        lower = mean - z * sd
        upper = mean + z * sd
        return mean, lower, upper


class NaiveModel(ForecastModel):
    """ナイーブ予測: 直前の値をそのまま将来へ引き伸ばす。"""

    name = "Naive"

    def fit(self, y) -> "NaiveModel":
        y = _to_array(y)
        if y.size == 0 or not _has_valid(y):
            raise ValueError("Naive: requires at least one valid observation")
        self._y = y
        self._filled = _filled(y)
        valid = self._filled[np.isfinite(self._filled)]
        self._last = float(valid[-1])
        return self

    def predict_in_sample(self) -> np.ndarray:
        f = self._filled
        fitted = np.full_like(f, np.nan)
        if f.size > 1:
            fitted[1:] = f[:-1]
        return fitted

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(max(0, int(horizon)), self._last)

    def residual_std(self) -> float:
        return _resid_std(self._filled, self.predict_in_sample())


class MovingAverageModel(ForecastModel):
    """移動平均予測: 直近 window 件の平均を将来へ引き伸ばす。"""

    name = "Moving Average"

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)

    def fit(self, y) -> "MovingAverageModel":
        y = _to_array(y)
        if y.size == 0 or not _has_valid(y):
            raise ValueError("MovingAverage: requires at least one valid observation")
        self._y = y
        self._filled = _filled(y)
        valid_count = int(np.isfinite(self._filled).sum())
        self._window_used = max(1, min(self.window, valid_count))
        return self

    def predict_in_sample(self) -> np.ndarray:
        f = self._filled
        s = pd.Series(f).rolling(self._window_used, min_periods=1).mean()
        return s.shift(1).to_numpy(dtype=float)

    def predict(self, horizon: int) -> np.ndarray:
        h = max(0, int(horizon))
        valid = self._filled[np.isfinite(self._filled)]
        if valid.size == 0:
            return np.full(h, np.nan)
        base = float(np.mean(valid[-self._window_used :]))
        return np.full(h, base)

    def residual_std(self) -> float:
        return _resid_std(self._filled, self.predict_in_sample())


class LinearTrendModel(ForecastModel):
    """線形トレンド予測: OLS で a + b*t を当てはめ、将来へ外挿する。"""

    name = "Linear Trend"

    def fit(self, y) -> "LinearTrendModel":
        y = _to_array(y)
        if y.size == 0 or not _has_valid(y):
            raise ValueError("LinearTrend: requires at least one valid observation")
        self._y = y
        self._n = y.size
        t = np.arange(self._n, dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 2:
            val = float(np.nanmean(y))
            self._a, self._b = val, 0.0
        else:
            slope, intercept = np.polyfit(t[mask], y[mask], 1)
            self._a, self._b = float(intercept), float(slope)
        return self

    def predict_in_sample(self) -> np.ndarray:
        t = np.arange(self._n, dtype=float)
        return self._a + self._b * t

    def predict(self, horizon: int) -> np.ndarray:
        h = max(0, int(horizon))
        t = np.arange(self._n, self._n + h, dtype=float)
        return self._a + self._b * t

    def residual_std(self) -> float:
        return _resid_std(self._y, self.predict_in_sample())
