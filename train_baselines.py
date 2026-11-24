import sys
from pathlib import Path

import pandas as pd
import torch

from evaluation.metrics import evaluate
from models.baseline.historical_avg import HistoricalAverageBaseline
from models.baseline.linear_regression import PerZoneLinearRegression
from models.baseline.seasonal_naive import SeasonalNaiveBaseline
from preprocessing.create_sequences import LOOKBACK, build_splits

RESULTS_DIR = Path("results/baselines")


@torch.no_grad()
def predict(model: torch.nn.Module, split: dict) -> torch.Tensor:
    inputs = [split["history"].split(512)]
    if isinstance(model, PerZoneLinearRegression):
        inputs.append(split["calendar"].split(512))

    return torch.cat([model(*batch) for batch in zip(*inputs)])


def main() -> None:
    torch.manual_seed(0)
    splits = build_splits()
    train, val = splits["train"], splits["val"]

    weeks_mae = {w: evaluate(predict(HistoricalAverageBaseline(weeks=w), val), val["target"])["mae"] for w in range(1, LOOKBACK // 168 + 1)}
    best_weeks = min(weeks_mae, key=weeks_mae.get)

    regression = PerZoneLinearRegression(train["target"].shape[1])
    regression.fit(train["history"], train["calendar"], train["target"], validation=(val["history"], val["calendar"], val["target"]))

    models = {
        "seasonal_naive": SeasonalNaiveBaseline(),
        "historical_average": HistoricalAverageBaseline(weeks=best_weeks),
        "linear_regression": regression,
    }
    split_names = ["train", "val", "test"] if "--test" in sys.argv else ["train", "val"]
    results = pd.DataFrame(
        {"model": name, "split": s, **evaluate(predict(model, splits[s]), splits[s]["target"])}
        for name, model in models.items()
        for s in split_names
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
