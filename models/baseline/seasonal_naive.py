import torch
from torch import nn


class SeasonalNaiveBaseline(nn.Module):
    def __init__(self, period: int = 168) -> None:
        super().__init__()
        self.period = period

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history[..., -self.period, :]
