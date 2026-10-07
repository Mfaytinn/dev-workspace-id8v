"""Random Toy splits and chronological air-quality splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split


def split_toy_data(
    x: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    x_train, x_pool, b_train, b_pool, y_train, y_pool = train_test_split(
        x,
        b,
        y,
        test_size=val_size + test_size,
        random_state=seed,
    )

    val_ratio = val_size / (val_size + test_size)
    x_val, x_rem, b_val, b_rem, y_val, y_rem = train_test_split(
        x_pool,
        b_pool,
        y_pool,
        test_size=1.0 - val_ratio,
        random_state=seed,
    )
    return (x_train, b_train, y_train), (x_val, b_val, y_val), (x_rem, b_rem, y_rem)


def split_cal_test_pool(
    x: np.ndarray,
    b: np.ndarray,
    y: np.ndarray,
    seed: int,
    test_size: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(x, b, y, test_size=test_size, random_state=seed)


def subsample_train_indices(
    n_train: int,
    frac: float,
    seed: int,
) -> np.ndarray:
    """Indices into a train split, randomly subsampled down to `frac` of rows.

    Deterministic given (n_train, frac, seed). `frac == 1.0` returns
    `np.arange(n_train)` unchanged. `n_keep` is rounded and clamped to
    `[1, n_train]`.
    """
    if float(frac) >= 1.0:
        return np.arange(int(n_train))
    n_keep = max(1, min(int(n_train), round(int(n_train) * float(frac))))
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.permutation(int(n_train))[:n_keep])


def reserve_temporal_holdout(
    df: pd.DataFrame,
    *,
    block_size_days: int,
    gap_hours: int,
    cal_frac: float,
    test_frac: float,
) -> tuple[int, dict]:
    """Reserve final calibration/test blocks outside all development origins."""
    data, blocks = _assign_calendar_blocks(df, block_size_days)
    rows = data.groupby("_block_id").size().reindex(blocks).to_numpy()
    prefix = np.concatenate(([0], np.cumsum(rows)))
    target = np.array([1.0 - cal_frac - test_frac, cal_frac, test_frac])
    best = None
    for cal_start in range(1, len(blocks) - 1):
        for test_start in range(cal_start + 1, len(blocks)):
            counts = np.array(
                [
                    prefix[cal_start],
                    prefix[test_start] - prefix[cal_start],
                    prefix[-1] - prefix[test_start],
                ]
            )
            key = (
                float(np.abs(counts / prefix[-1] - target).sum()),
                -int(counts.min()),
                cal_start,
                test_start,
            )
            if best is None or key < best[0]:
                best = (key, cal_start, test_start)
    if best is None:
        raise ValueError(
            "At least three observed temporal blocks are needed to reserve a holdout"
        )
    _, cal_start, test_start = best
    cal_blocks, test_blocks = blocks[cal_start:test_start], blocks[test_start:]
    reserved_start = int(cal_blocks[0])
    reserved = data[data["_block_id"] >= reserved_start].copy()
    reserved = _apply_calendar_gap(
        reserved, gap_hours=gap_hours, left_boundary_blocks=[int(cal_blocks[-1])]
    )
    parts = {
        "cal": reserved[reserved["_block_id"].isin(cal_blocks)],
        "test": reserved[reserved["_block_id"].isin(test_blocks)],
    }
    return reserved_start, {
        "protocol": "reserved_temporal_holdout",
        "reserved_cal_start_block": reserved_start,
        "minimum_gap_hours": gap_hours,
        "splits": {
            name: {
                "rows": len(frame),
                "blocks": int(frame["_block_id"].nunique()),
                "first_timestamp": frame["date"].min().isoformat(),
                "last_timestamp": frame["date"].max().isoformat(),
            }
            for name, frame in parts.items()
        },
    }


def build_random_split_indices(
    n_samples: int,
    n_splits: int,
    test_size: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    return [
        (cal_idx, test_idx)
        for cal_idx, test_idx in splitter.split(np.arange(n_samples))
    ]


def _assign_calendar_blocks(
    df: pd.DataFrame,
    block_size_days: int,
) -> tuple[pd.DataFrame, np.ndarray]:

    data = df.sort_values("date").reset_index(drop=True).copy()
    t_min = data["date"].min()
    block_size = pd.Timedelta(days=block_size_days)
    offsets = (data["date"] - t_min).dt.total_seconds()
    block_id = (offsets // block_size.total_seconds()).astype(int)
    data["_block_id"] = block_id
    unique_blocks = np.array(sorted(data["_block_id"].unique()))
    return data, unique_blocks


def _apply_calendar_gap(
    data: pd.DataFrame,
    gap_hours: int,
    left_boundary_blocks: list[int],
) -> pd.DataFrame:
    if gap_hours == 0:
        return data

    gap = np.timedelta64(gap_hours, "h")
    keep_mask = np.ones(len(data), dtype=bool)
    ts = data["date"].to_numpy(dtype="datetime64[ns]")

    block_end: dict[int, np.datetime64] = {}
    for bid in np.array(sorted(data["_block_id"].unique())):
        rows = data[data["_block_id"] == bid]
        block_end[int(bid)] = rows["date"].max().to_datetime64()

    for left in left_boundary_blocks:
        boundary = block_end[int(left)]
        keep_mask &= ~((ts >= boundary - gap) & (ts < boundary + gap))

    return data[keep_mask].copy()
