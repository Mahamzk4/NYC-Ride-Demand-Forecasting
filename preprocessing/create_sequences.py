from pathlib import Path

import numpy as np
import pandas as pd
import torch

MATRIX_FILE = Path("data/processed/hourly_zone_matrix_clean.csv")
SPLITS = {
    "train": ("2024-01-01 00:00", "2025-06-30 23:00"),
    "val": ("2025-07-01 00:00", "2025-09-30 23:00"),
    "test": ("2025-10-01 00:00", "2025-12-31 23:00"),
}
LOOKBACK = 672


def load_matrix(path: Path = MATRIX_FILE) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col="hour", parse_dates=["hour"])
    matrix.columns = matrix.columns.astype(int)
    assert list(matrix.columns) == list(range(1, 264))
    assert matrix.index.equals(pd.date_range(matrix.index[0], matrix.index[-1], freq="h"))
    
    return matrix


def calendar_features(hours: pd.DatetimeIndex) -> torch.Tensor:
    hour, day = hours.hour.to_numpy(), hours.day_of_week.to_numpy()
    features = np.stack([np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * day / 7), np.cos(2 * np.pi * day / 7)], axis=1)

    return torch.tensor(features, dtype=torch.float32)


def build_splits(matrix: pd.DataFrame | None = None, splits: dict = SPLITS, lookback: int = LOOKBACK) -> dict:
    matrix = load_matrix() if matrix is None else matrix
    series = torch.tensor(matrix.to_numpy(), dtype=torch.float32)
    windows = series.unfold(0, lookback, 1).transpose(1, 2)
    calendar = calendar_features(matrix.index)

    result = {}
    for name, (start, end) in splits.items():
        first = max(matrix.index.get_loc(pd.Timestamp(start)), lookback)
        last = matrix.index.get_loc(pd.Timestamp(end)) + 1
        result[name] = {
            "history": windows[first - lookback : last - lookback],
            "calendar": calendar[first:last],
            "target": series[first:last],
            "hours": matrix.index[first:last],
        }
        
    return result
