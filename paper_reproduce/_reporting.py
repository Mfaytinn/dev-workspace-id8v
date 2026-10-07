"""Shared summaries for experiment result tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

MEASURES = (
    "coverage_primary",
    "coverage_reference_row",
    "coverage_sensor_row",
    "coverage_reference_timestamp",
    "coverage_sensor_timestamp",
    "width",
)


def summarize_results(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool origins within seeds, then summarize independent seeds."""
    keys = ["dataset", "method", "held_out_sensor"]
    per_seed = []
    for group_key, group in raw.groupby([*keys, "seed"], sort=False):
        temporal = group["origin"].notna().any()
        row = dict(zip([*keys, "seed"], group_key, strict=True))
        row["anchor_sensors"] = ", ".join(dict.fromkeys(group["anchor_sensor"]))
        for measure in MEASURES:
            values = group[measure].to_numpy(dtype=float)
            if temporal:
                primary_row = (
                    measure == "coverage_primary"
                    and "coverage_unit" in group
                    and group["coverage_unit"].eq("per_location").all()
                )
                weights = group[
                    (
                        "test_timestamps"
                        if "timestamp" in measure
                        or (measure == "coverage_primary" and not primary_row)
                        else "test_rows"
                    )
                ].to_numpy(dtype=float)
                row[measure] = float(np.average(values, weights=weights))
            else:
                row[measure] = float(values.mean())
        per_seed.append(row)
    seed_frame = pd.DataFrame(per_seed)
    summary = seed_frame.groupby(keys, sort=False)[list(MEASURES)].agg(["mean", "std"])
    summary.columns = [f"{measure}_{stat}" for measure, stat in summary.columns]
    summary["n_seeds"] = seed_frame.groupby(keys, sort=False).size()
    summary = summary.reset_index()
    anchors = (
        seed_frame.groupby(keys, sort=False)["anchor_sensors"]
        .agg(
            lambda values: ", ".join(
                dict.fromkeys(
                    sensor.strip() for value in values for sensor in value.split(",")
                )
            )
        )
        .reset_index()
    )
    return seed_frame, summary.merge(anchors, on=keys, validate="one_to_one")
