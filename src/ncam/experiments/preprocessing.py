"""Temporal features, training-only scaling, and observation-space transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def add_temporal_features(
    df: pd.DataFrame,
    add_hour: bool = True,
    add_day_of_year: bool = True,
    add_day_of_week: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Add sin/cos-encoded temporal features derived from the 'date' column.

    Returns the augmented DataFrame and a list of new column names added.
    """
    new_cols: list[str] = []
    dt = pd.to_datetime(df["date"])

    if add_hour:
        hour_frac = dt.dt.hour + dt.dt.minute / 60.0
        df = df.copy()
        df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24.0).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24.0).astype(np.float32)
        new_cols += ["hour_sin", "hour_cos"]

    if add_day_of_year:
        doy = dt.dt.dayofyear.astype(np.float32)
        if "hour_sin" not in df.columns:
            df = df.copy()
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
        new_cols += ["doy_sin", "doy_cos"]

    if add_day_of_week:
        dow = dt.dt.dayofweek.astype(np.float32)
        if "hour_sin" not in df.columns and "doy_sin" not in df.columns:
            df = df.copy()
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
        new_cols += ["dow_sin", "dow_cos"]

    return df, new_cols


def aggregate_time_buckets(df: pd.DataFrame, sampling_minutes: int) -> pd.DataFrame:

    value_cols = [c for c in df.columns if c not in {"location_id", "date"}]

    freq = f"{sampling_minutes}min"
    bucketed = df.sort_values(["location_id", "date"]).copy()
    bucketed["_time_bucket"] = bucketed["date"].dt.floor(freq)
    return (
        bucketed.groupby(["location_id", "_time_bucket"], as_index=False)[value_cols]
        .mean()
        .rename(columns={"_time_bucket": "date"})
        .sort_values(["location_id", "date"])
        .reset_index(drop=True)
    )


def fit_feature_scaler(
    x_train: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Training feature mean and sample deviation; constant columns have scale one."""
    x = torch.as_tensor(x_train, dtype=torch.float32)
    if x.ndim != 2 or len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError("Scaling requires at least two finite training feature rows")
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


def fit_coordinate_bounds(
    x_train: np.ndarray, feature_names: list[str], *, enabled: bool
) -> dict[str, list[float]]:
    """Fit geographic feature bounds using only rows in the training partition."""
    if not enabled:
        return {}
    bounds = {}
    for name in ("latitude", "longitude"):
        if name in feature_names:
            values = x_train[:, feature_names.index(name)]
            if not np.isfinite(values).all():
                raise ValueError(f"Coordinate feature {name} must be finite")
            bounds[name] = [float(values.min()), float(values.max())]
    return bounds


def clip_coordinate_features(
    x: np.ndarray,
    feature_names: list[str],
    bounds: dict[str, list[float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Clamp geographic feature columns and report clipped row counts."""
    clipped = x.copy()
    counts = {}
    for name, (lower, upper) in bounds.items():
        index = feature_names.index(name)
        values = x[:, index]
        outside = (values < lower) | (values > upper)
        counts[name] = int(outside.sum())
        clipped[:, index] = np.clip(values, lower, upper)
    return clipped, counts


def resolve_training_batch_size(cfg: dict, n_train: int) -> int:
    """Apply one data-size rule for both air-quality training entrypoints."""
    value = cfg["batch_size"]
    if value != "auto":
        batch_size = int(value)
    else:
        updates = int(cfg["target_updates_per_epoch"])
        minimum = int(cfg["min_batch_size"])
        maximum = int(cfg["max_batch_size"])
        batch_size = min(max((n_train + updates - 1) // updates, minimum), maximum)
    cfg["batch_size"] = batch_size
    return batch_size


def compute_log_floors_from_train(
    b_train_raw: torch.Tensor, sensor_names: list[str]
) -> torch.Tensor:
    floors = []
    for j, _name in enumerate(sensor_names):
        col = b_train_raw[:, j].float()
        mask = col > 0.0
        pos = col[mask]
        if pos.numel() == 0:
            raise ValueError(f"No positive training observations for sensor {_name}")
        floors.append(float(pos.min().item()))

    return torch.tensor(floors, dtype=torch.float32, device=b_train_raw.device)


def apply_obs_transform_log(
    b_values: torch.Tensor,
    log_floors: torch.Tensor,
) -> torch.Tensor:
    b = b_values.float()

    floors = log_floors.float().to(b.device)
    return torch.log(torch.maximum(b, floors.reshape(1, -1)))
