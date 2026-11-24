import torch


def evaluate(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = predictions - targets

    return {
        "mae": error.abs().mean().item(),
        "rmse": error.pow(2).mean().sqrt().item(),
        "wape": (error.abs().sum() / targets.sum()).item(),
    }
