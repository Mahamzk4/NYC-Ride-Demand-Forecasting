import copy

import torch
from torch import nn
from torch.nn import functional as F


class PerZoneLinearRegression(nn.Module):
    def __init__(self, num_zones: int, history_length: int = 336, calendar_features: int = 4) -> None:
        super().__init__()
        self.history_length = history_length
        self.history_weight = nn.Parameter(torch.zeros(num_zones, history_length))
        self.calendar_weight = nn.Parameter(torch.zeros(num_zones, calendar_features))
        self.bias = nn.Parameter(torch.zeros(num_zones))
        self.register_buffer("scale", torch.ones(num_zones))

    def _scaled(self, history: torch.Tensor, target_calendar: torch.Tensor) -> torch.Tensor:
        recent = history[:, -self.history_length :, :] / self.scale

        return (recent * self.history_weight.T).sum(dim=1) + target_calendar @ self.calendar_weight.T + self.bias

    def forward(self, history: torch.Tensor, target_calendar: torch.Tensor) -> torch.Tensor:
        return (self._scaled(history, target_calendar) * self.scale).clamp_min(0)

    def fit(self, history, target_calendar, targets, *, epochs=100, batch_size=256, learning_rate=3e-3, validation=None, patience=10):
        self.scale = targets.mean(dim=0).clamp_min(1)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        best_loss, best_state, bad_epochs = float("inf"), None, 0

        for _ in range(epochs):
            for batch in torch.randperm(len(history)).split(batch_size):
                loss = F.mse_loss(self._scaled(history[batch], target_calendar[batch]), targets[batch] / self.scale)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if validation is None:
                continue

            val_history, val_calendar, val_targets = validation

            with torch.no_grad():
                predictions = torch.cat([self._scaled(h, c) for h, c in zip(val_history.split(1024), val_calendar.split(1024))])
                val_loss = F.mse_loss(predictions, val_targets / self.scale).item()

            if val_loss < best_loss:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(self.state_dict()), 0
            else:
                bad_epochs += 1
                
            if bad_epochs >= patience:
                break

        if best_state is not None:
            self.load_state_dict(best_state)

        return self
