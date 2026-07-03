"""data.py の単体テスト。"""

import numpy as np
import pytest

from forecasting_lab.data import DATASET_NAMES, load_dataset, synthetic_with_missing


def test_all_datasets_load():
    for name in DATASET_NAMES:
        df = load_dataset(name, n=60, seed=1)
        assert {"date", "value"} == set(df.columns)
        assert len(df) == 60


def test_missing_dataset_contains_nan():
    df = synthetic_with_missing(n=200, seed=0, missing_frac=0.2)
    assert df["value"].isna().any()


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("does_not_exist", n=60)


def test_too_short_raises():
    with pytest.raises(ValueError):
        load_dataset(DATASET_NAMES[0], n=1)


def test_reproducible_with_seed():
    a = load_dataset(DATASET_NAMES[0], n=60, seed=42)
    b = load_dataset(DATASET_NAMES[0], n=60, seed=42)
    np.testing.assert_allclose(a["value"].to_numpy(), b["value"].to_numpy())
