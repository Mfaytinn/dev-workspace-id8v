"""Named data passed between preparation, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import pandas as pd
    import torch


class PerformanceRow(NamedTuple):
    method: str
    rmse: float
    mae: float


class TensorSplit(NamedTuple):
    """Features (N,D), model-space sensors (N,L), raw targets (N,)."""

    x: torch.Tensor
    sensors: torch.Tensor
    targets: torch.Tensor


class SplitFrames(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    cal: pd.DataFrame
    test: pd.DataFrame


@dataclass
class PreparedFold:
    config: dict
    train: TensorSplit
    val: TensorSplit
    cal: TensorSplit
    test: TensorSplit
    frames: SplitFrames
    pool: TensorSplit | None = None
