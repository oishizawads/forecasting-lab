"""models.py の単体テスト。"""

import numpy as np
import pytest

from forecasting_lab.models import LinearTrendModel, MovingAverageModel, NaiveModel


def test_naive_predict_uses_last_value():
    m = NaiveModel().fit([1, 2, 3])
    assert np.allclose(m.predict(3), [3, 3, 3])


def test_naive_in_sample_shifted():
    fitted = NaiveModel().fit([1, 2, 3]).predict_in_sample()
    assert np.isnan(fitted[0])
    assert np.allclose(fitted[1:], [1, 2])


def test_moving_average_predict_is_window_mean():
    m = MovingAverageModel(window=2).fit([1, 2, 4, 8])
    assert np.allclose(m.predict(2), [6, 6])  # mean(4,8)=6


def test_moving_average_window_capped_to_valid():
    m = MovingAverageModel(window=10).fit([1, 2, 3])
    assert np.allclose(m.predict(1), [2.0])  # mean(1,2,3)=2


def test_linear_trend_recovers_line():
    m = LinearTrendModel().fit([1, 2, 3, 4])
    assert np.allclose(m.predict_in_sample(), [1, 2, 3, 4])
    assert np.allclose(m.predict(2), [5, 6])


def test_predict_interval_shapes_and_order():
    m = NaiveModel().fit([1, 2, 3, 4])
    mean, lo, hi = m.predict_interval(3, z=1.96)
    assert mean.shape == (3,)
    assert np.all(lo <= mean) and np.all(mean <= hi)


def test_empty_input_raises():
    for cls in (NaiveModel, MovingAverageModel, LinearTrendModel):
        with pytest.raises(ValueError):
            cls().fit([]) if cls is not MovingAverageModel else cls(window=2).fit([])


def test_all_nan_raises():
    arr = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        NaiveModel().fit(arr)
    with pytest.raises(ValueError):
        MovingAverageModel(window=2).fit(arr)
    with pytest.raises(ValueError):
        LinearTrendModel().fit(arr)


def test_missing_values_do_not_crash_naive():
    m = NaiveModel().fit([1, 2, np.nan, 4])
    assert np.allclose(m.predict(2), [4, 4])  # 最後の有効値


def test_missing_values_do_not_crash_linear_trend():
    m = LinearTrendModel().fit([1, 2, np.nan, 4, 5])
    fitted = m.predict_in_sample()
    assert fitted.shape == (5,)
    pred = m.predict(2)
    assert pred.shape == (2,)


def test_moving_average_window_must_be_positive():
    with pytest.raises(ValueError):
        MovingAverageModel(window=0)
