"""Shared training-only preparation for Toy and calendar experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ncam.experiments.data import (
    extract_arrays_from_df,
    load_cairsense_data,
    load_seneurcity_data,
)
from ncam.experiments.preprocessing import (
    add_temporal_features,
    aggregate_time_buckets,
    apply_obs_transform_log,
    clip_coordinate_features,
    compute_log_floors_from_train,
    fit_coordinate_bounds,
    fit_feature_scaler,
    resolve_training_batch_size,
)
from ncam.experiments.records import (
    PreparedFold,
    SplitFrames,
    TensorSplit,
)
from ncam.experiments.splits import (
    split_cal_test_pool,
    split_toy_data,
    subsample_train_indices,
)

DATASET_COLUMNS = {
    "toy_spatial": (
        ["sin_lat", "cos_lat", "sin_lon", "cos_lon"],
        ["sensor_0", "sensor_1", "sensor_2"],
        "y_true",
    ),
    "seneurcity": (
        [
            "latitude",
            "longitude",
            "SHT31TE",
            "SHT31HE",
            "Absolute_humidity",
            "Td_deficit",
            "BMP280",
        ],
        ["OPCN3PM25", "5325CAT", "5325CST"],
        "Ref.PM2.5",
    ),
    "cairsense": (
        ["TEMP", "RHAMB", "WS", "WD"],
        ["AirAssure1", "AirAssure2", "AirAssure3"],
        "SoC",
    ),
}
DATASET_SENSORS = {
    dataset: tuple(columns[1]) for dataset, columns in DATASET_COLUMNS.items()
}
TEMPORAL_COLUMNS = {"hour_sin", "hour_cos", "doy_sin", "doy_cos", "dow_sin", "dow_cos"}


def load_dataset(
    dataset: str, config: dict, *, saved_config: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Apply the dataset's column selection, row filtering, and hourly means."""
    default_x, default_b, y_col = DATASET_COLUMNS[dataset]
    cfg = dict(config)
    cfg["x_cols"] = list(cfg.get("x_cols") or default_x)
    cfg["b_cols"] = list(cfg.get("b_cols") or default_b)
    cfg["y_col"] = y_col
    if dataset == "toy_spatial":
        cfg.setdefault("anchor_sensor", cfg["b_cols"][int(cfg["anchor_idx"])])
    if saved_config is not None:
        cfg["x_cols"] = [c for c in saved_config["x_cols"] if c not in TEMPORAL_COLUMNS]
        cfg["b_cols"] = list(saved_config["b_cols"])
    x_cols, b_cols = cfg["x_cols"], cfg["b_cols"]
    require_reference = saved_config is None and not bool(
        cfg.get("sensor_only_train", False)
    )
    if dataset == "toy_spatial":
        frame = pd.read_csv(cfg["data_path"])
        columns = [*x_cols, *b_cols, y_col]
        frame = (
            frame[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
        )
    else:
        if dataset == "cairsense":
            frame = load_cairsense_data(
                cfg["data_path"],
                x_cols,
                b_cols,
                y_col,
                require_reference=require_reference,
            )
        else:
            frame = load_seneurcity_data(
                cfg["data_dir"],
                cfg["pattern"],
                x_cols,
                b_cols,
                y_col,
                require_reference=require_reference,
            )
        frame = aggregate_time_buckets(frame, sampling_minutes=60)
        if bool(cfg["temporal_features"]):
            frame, extra = add_temporal_features(
                frame,
                add_day_of_year=dataset != "cairsense"
                or not bool(cfg.get("omit_day_of_year_features", False)),
            )
            cfg["x_cols"] = [*x_cols, *extra]
    if saved_config is not None and cfg["x_cols"] != saved_config["x_cols"]:
        raise ValueError("Reconstructed features differ from the saved checkpoint")
    return frame, cfg


def toy_split_frames(
    frame: pd.DataFrame, cfg: dict
) -> tuple[SplitFrames, pd.DataFrame]:
    """Keep the original random split membership, including the held-out pool."""
    indices = np.arange(len(frame))
    train, val, pool = split_toy_data(
        indices,
        indices,
        indices,
        float(cfg["val_size"]),
        float(cfg["test_size"]),
        int(cfg["seed"]),
    )
    cal_idx, test_idx, *_ = split_cal_test_pool(
        pool[0], pool[0], pool[0], seed=int(cfg["seed"]), test_size=0.5
    )
    return (
        SplitFrames(
            frame.iloc[train[0]],
            frame.iloc[val[0]],
            frame.iloc[cal_idx],
            frame.iloc[test_idx],
        ),
        frame.iloc[pool[0]],
    )


def prepare_fold(
    frames: SplitFrames,
    cfg: dict,
    device: str,
    *,
    pool: pd.DataFrame | None = None,
    subsample: bool = False,
    fitted: bool = False,
) -> PreparedFold:
    """Fit scaling and log floors on training rows, then transform all splits.

    Features and sensors are float32. Targets remain in raw observation space.
    Training and validation tensors stay on CPU until loaders move them.
    """
    if subsample and float(cfg.get("train_subsample_frac", 1.0)) < 1.0:
        indices = subsample_train_indices(
            len(frames.train), float(cfg["train_subsample_frac"]), int(cfg["seed"])
        )
        frames = frames._replace(train=frames.train.iloc[indices])
    arrays = [
        extract_arrays_from_df(f, cfg["x_cols"], cfg["b_cols"], cfg["y_col"])
        for f in frames
    ]
    train_x, train_b, _ = arrays[0]
    if fitted:
        coordinate_bounds = cfg.get("coordinate_bounds", {})
        mean = torch.as_tensor(cfg["x_norm_mean"], dtype=torch.float32)
        std = torch.as_tensor(cfg["x_norm_std"], dtype=torch.float32)
    else:
        coordinate_bounds = fit_coordinate_bounds(
            train_x, cfg["x_cols"], enabled=bool(cfg.get("clip_coordinates", False))
        )
        mean, std = fit_feature_scaler(train_x)
    fold_cfg = dict(cfg)
    fold_cfg.pop("coordinate_bounds", None)
    fold_cfg.pop("coordinate_clipping", None)
    if bool(cfg.get("clip_coordinates", False)):
        fold_cfg["coordinate_bounds"] = coordinate_bounds
    coordinate_clipping = {}
    if cfg.get("model") in {"anchored", "unanchored"}:
        if cfg.get("prior_mode", "contextual") != "contextual":
            raise ValueError("NCAM requires a contextual prior")
        if cfg.get("gain_mode", "learned") != "learned":
            raise ValueError("Anchored NCAM requires learned non-anchor gains")
        fold_cfg.update(prior_mode="contextual", gain_mode="learned")
    fold_cfg["x_norm_mean"] = mean.tolist()
    fold_cfg["x_norm_std"] = std.tolist()
    resolve_training_batch_size(fold_cfg, len(frames.train))
    floors = None
    if cfg["obs_transform"] == "log":
        floors = (
            torch.as_tensor(cfg["log_floor_per_sensor"], dtype=torch.float32)
            if fitted
            else compute_log_floors_from_train(torch.as_tensor(train_b), cfg["b_cols"])
        )
        fold_cfg["log_floor_per_sensor"] = floors.tolist()
    else:
        fold_cfg.pop("log_floor_per_sensor", None)

    def transform(values, target_device, partition):
        x, b, y = values
        x, counts = clip_coordinate_features(x, cfg["x_cols"], coordinate_bounds)
        if coordinate_bounds:
            coordinate_clipping[partition] = counts
        x = (torch.as_tensor(x, dtype=torch.float32) - mean) / std
        b = torch.as_tensor(b, dtype=torch.float32)
        b = apply_obs_transform_log(b, floors) if floors is not None else b
        if cfg["obs_transform"] == "log1p":
            if torch.any(b < 0):
                raise ValueError("log1p PM observations must be nonnegative")
            b = torch.log1p(b)
        return TensorSplit(
            x.to(target_device),
            b.to(target_device),
            torch.as_tensor(y, dtype=torch.float32, device=target_device).ravel(),
        )

    splits = [
        transform(values, "cpu" if i < 2 else device, name)
        for i, (name, values) in enumerate(zip(frames._fields, arrays, strict=True))
    ]
    prepared_pool = (
        transform(
            extract_arrays_from_df(pool, cfg["x_cols"], cfg["b_cols"], cfg["y_col"]),
            device,
            "pool",
        )
        if pool is not None
        else None
    )
    if bool(cfg.get("clip_coordinates", False)):
        fold_cfg["coordinate_clipping"] = coordinate_clipping
    return PreparedFold(fold_cfg, *splits, frames, prepared_pool)


def prepare_toy_fold(
    df: pd.DataFrame, cfg: dict, device: str, *, fitted=False
) -> PreparedFold:
    """Apply the synthetic dataset's random splits and training preprocessing."""
    frames, pool = toy_split_frames(df, cfg)
    return prepare_fold(frames, cfg, device, pool=pool, subsample=True, fitted=fitted)
