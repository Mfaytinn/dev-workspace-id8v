"""Calendar windows shared by retrospective real-data fold comparisons."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pandas as pd

from ncam.experiments.records import SplitFrames


def calendar_fold(
    frame: pd.DataFrame,
    test_end,
    *,
    train_days=56,
    val_days=7,
    cal_days=14,
    test_days=7,
    gap_hours=36,
    min_train_rows=600,
    min_val_rows=120,
    min_cal_timestamps=119,
    min_test_timestamps=24,
    min_observed_fraction=0.8,
    reference_col=None,
):
    """Build four past-only windows with explicit gaps and whole timestamps.

    test_end is exclusive. No row values or prediction errors choose membership.
    Unlike row-count windows, calendar durations do not grow with device count.
    """
    durations = (train_days, val_days, cal_days, test_days)
    if any(not np.isfinite(d) or d <= 0 for d in durations):
        raise ValueError("Window durations must be positive")
    if any(not float(days * 24).is_integer() for days in durations):
        raise ValueError("Window durations must align to whole hours")
    if not np.isfinite(gap_hours) or gap_hours < 0 or not float(gap_hours).is_integer():
        raise ValueError("Gap must be a nonnegative whole number of hours")
    if not np.isfinite(min_observed_fraction) or not 0 < min_observed_fraction <= 1:
        raise ValueError("Minimum observed fraction must lie in (0, 1]")
    end = pd.Timestamp(test_end)
    if end != end.floor("h"):
        raise ValueError("Calendar endpoints must align to whole hours")
    gap = pd.Timedelta(hours=gap_hours)
    parts = []
    windows = {}
    for name, days in reversed(list(zip(SplitFrames._fields, durations, strict=True))):
        start = end - pd.Timedelta(days=days)
        part = frame.loc[(frame.date >= start) & (frame.date < end)].copy()
        parts.append(part)
        windows[name] = {"start": start.isoformat(), "end_exclusive": end.isoformat()}
        end = start - gap
    folds = SplitFrames(*reversed(parts))
    if reference_col is not None:
        if reference_col not in folds.cal or reference_col not in folds.test:
            raise ValueError(f"Reference column is missing: {reference_col}")
        folds = folds._replace(
            cal=folds.cal.dropna(subset=[reference_col]).copy(),
            test=folds.test.dropna(subset=[reference_col]).copy(),
        )
    if folds.train.date.min() > pd.Timestamp(windows["train"]["start"]) + pd.Timedelta(
        hours=24
    ):
        raise ValueError("Training window begins before usable dataset observations")
    if len(folds.train) < min_train_rows or len(folds.val) < min_val_rows:
        raise ValueError("Insufficient training or validation rows in calendar window")
    val_hours = folds.val.date.dt.floor("h").nunique()
    cal_hours = folds.cal.date.dt.floor("h").nunique()
    test_hours = folds.test.date.dt.floor("h").nunique()
    expected_val_hours = round(val_days * 24)
    expected_cal_hours = round(cal_days * 24)
    expected_test_hours = round(test_days * 24)
    required_val_hours = math.ceil(expected_val_hours * min_observed_fraction)
    required_cal_hours = max(
        int(min_cal_timestamps),
        math.ceil(expected_cal_hours * min_observed_fraction),
    )
    required_test_hours = max(
        int(min_test_timestamps),
        math.ceil(expected_test_hours * min_observed_fraction),
    )
    if val_hours < required_val_hours:
        raise ValueError("Insufficient observed validation timestamps")
    if cal_hours < required_cal_hours:
        raise ValueError("Insufficient observed calibration timestamps")
    if test_hours < required_test_hours:
        raise ValueError("Insufficient observed test timestamps")
    for left, right in pairwise(folds):
        if right.date.min() - left.date.max() < gap:
            raise ValueError("Observed splits violate the time gap")
    return folds, windows


def window_statistics(frames, sensors, reference):
    """Raw-unit summaries including device representation and reference RMSE."""
    rows = []
    training_devices = set(frames.train.location_id)
    calibration_devices = set(frames.cal.location_id)
    for name, frame in zip(frames._fields, frames, strict=True):
        devices = set(frame.location_id)
        counts = frame.groupby("date").size()
        for sensor in (*sensors, reference):
            values = frame[sensor].to_numpy(dtype=float)
            target = frame[reference].to_numpy(dtype=float)
            finite = np.isfinite(values) & np.isfinite(target)
            residual = values[finite] - target[finite]
            rows.append(
                {
                    "partition": name,
                    "sensor": sensor,
                    "rows": len(frame),
                    "timestamps": frame.date.nunique(),
                    "devices": len(devices),
                    "devices_absent_training": len(devices - training_devices),
                    "devices_absent_calibration": len(devices - calibration_devices),
                    "median_devices_per_timestamp": float(counts.median()),
                    "first": frame.date.min().isoformat(),
                    "last": frame.date.max().isoformat(),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "nonpositive_rows": int((values <= 0).sum()),
                    "rmse_to_reference": float(np.sqrt(np.mean(residual**2))),
                    "mean_bias_to_reference": float(np.mean(residual)),
                }
            )
    return rows
