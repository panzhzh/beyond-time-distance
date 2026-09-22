"""Portable array input and JSON output; paths are supplied by the caller."""
import json
from pathlib import Path

import numpy as np
import torch

from .base import ForecastInputs


def load_forecasts(path, device, *, targets=True):
    with np.load(path, allow_pickle=False) as data:
        history = torch.as_tensor(data["history"], dtype=torch.float64, device=device)
        quantiles = torch.as_tensor(data["quantiles"], dtype=torch.float64, device=device)
        if "scale" in data:
            scale = torch.as_tensor(data["scale"], dtype=torch.float64, device=device)
        else:
            quartiles = torch.quantile(history, history.new_tensor([.25, .75]), dim=1)
            scale = (quartiles[1] - quartiles[0]).clamp_min(1e-6)
        target = torch.as_tensor(data["target"], dtype=torch.float64, device=device) if targets else None
    inputs = ForecastInputs(history, quantiles, scale)
    inputs.validate(quantiles.shape[1])
    if targets and (target.shape != quantiles.shape[:2] or not bool(torch.isfinite(target).all())):
        raise ValueError("Expected finite N x H targets")
    return inputs, target


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
