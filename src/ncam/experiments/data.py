"""Dataset loading and sensor tensor batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_arrays_from_df(
    df: pd.DataFrame,
    x_cols: list[str],
    b_cols: list[str],
    y_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract arrays; invalid inputs must never become fabricated zero readings."""
    x = np.asarray(df[x_cols], dtype=np.float32)
    b = np.asarray(df[b_cols], dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(b).all():
        raise ValueError("Features and input sensor readings must be finite")
    y = np.asarray(df[y_col], dtype=np.float32).ravel()
    return x, b, y


def load_seneurcity_data(
    data_dir: str,
    pattern: str,
    x_cols: list[str],
    b_cols: list[str],
    y_col: str,
    require_reference: bool = True,
) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob(pattern))

    frames: list[pd.DataFrame] = []
    for file_path in files:
        location_id = file_path.name.split("_")[1].split(".")[0]
        frame = pd.read_csv(file_path)
        frame["date"] = pd.to_datetime(frame["date"])
        keep_cols = x_cols + b_cols + [y_col, "date"]
        frame = frame[keep_cols].copy()
        for col in x_cols + b_cols + [y_col]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        required = x_cols + b_cols + ["date"]
        if require_reference:
            required.append(y_col)
        frame = frame.dropna(subset=required).reset_index(drop=True)
        if len(frame) == 0:
            continue
        frame["location_id"] = location_id
        frames.append(frame)

    if not frames:
        raise ValueError("No usable SensEURCity sensor observations")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(frames)} files.")
    return combined


def load_cairsense_data(
    data_path: str,
    x_cols: list[str],
    b_cols: list[str],
    y_col: str,
    require_reference: bool = True,
) -> pd.DataFrame:
    """Load CAIRSENSE data from a single Excel file."""
    df = pd.read_excel(data_path)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"])
    keep_cols = x_cols + b_cols + [y_col, "date"]
    df = df[keep_cols].copy()
    for col in x_cols + b_cols + [y_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    required = x_cols + b_cols + ["date"]
    if require_reference:
        required.append(y_col)
    df = df.dropna(subset=required).reset_index(drop=True)
    df["location_id"] = "cairsense"
    print(f"Loaded {len(df)} rows from {data_path}.")
    return df
