"""metrics.py の単体テスト。"""

import math

import numpy as np
import pytest

from forecasting_lab.metrics import evaluate, mae, rmse


def test_mae_perfect():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_basic():
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx((1 + 0 + 1) / 3)


def test_rmse_basic():
    # 差分 [0, -1] -> 二乗平均 0.5 -> sqrt = 1/sqrt(2)
    assert rmse([1, 2], [1, 3]) == pytest.approx(1.0 / math.sqrt(2))


def test_metrics_ignore_nan_pairs():
    # 中央のペアは予測側が NaN -> 評価から除外される
    assert mae([1, 2, 3], [1, np.nan, 3]) == 0.0
    assert rmse([1, 2, 3], [1, np.nan, 3]) == 0.0


def test_metrics_all_invalid_returns_nan():
    assert math.isnan(mae([np.nan, np.nan], [1, 2]))
    assert math.isnan(rmse([np.nan], [np.nan]))


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
    with pytest.raises(ValueError):
        rmse([1, 2], [1, 2, 3])


def test_evaluate_dict_keys():
    res = evaluate([1, 2, 3], [1, 2, 4])
    assert set(res.keys()) == {"MAE", "RMSE"}
    assert res["RMSE"] == pytest.approx(1.0 / math.sqrt(3))
