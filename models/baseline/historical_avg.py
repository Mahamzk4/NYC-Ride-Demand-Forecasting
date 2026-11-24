import torch
from torch import nn


class HistoricalAverageBaseline(nn.Module):
    def __init__(self, period: int = 168, weeks: int = 4) -> None:
        super().__init__()
        self.period = period
        self.weeks = weeks

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.shape[-2] < self.period * self.weeks:
            raise ValueError(f"history must contain at least {self.period * self.weeks} time steps")
            
        return torch.stack([history[..., -self.period * week, :] for week in range(1, self.weeks + 1)]).mean(dim=0)
