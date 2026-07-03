"""forecasting-lab: 時系列予測モデルの比較と評価モジュール。"""

from .data import DATASET_NAMES, load_csv, load_dataset
from .metrics import evaluate, mae, rmse
from .models import (
    ForecastModel,
    LinearTrendModel,
    MovingAverageModel,
    NaiveModel,
)

__all__ = [
    "DATASET_NAMES",
    "ForecastModel",
    "LinearTrendModel",
    "MovingAverageModel",
    "NaiveModel",
    "evaluate",
    "load_csv",
    "load_dataset",
    "mae",
    "rmse",
]

__version__ = "0.1.0"
