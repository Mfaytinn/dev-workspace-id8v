"""Reproducible SenseurCity QA/QC filtering pipeline.

This script applies the QA/QC flag logic documented in the SenseurCity paper and
stores intermediate datasets at every stage.

Paper linkage used in this implementation:
- docs/seneurcity.md:346-427 (Data collection and data flagging; code availability)
- docs/seneurcity.md:348-351, 395-396, 411-413 (flag labels and meanings)
- docs/seneurcity.md:379-391 (Table 11 filtering parameters)

Important implementation note:
The paper states that filtering is carried out with Filter_Sensor_Data() from
Functions4ASE.R, and the resulting flags are already included in the published
CSV files. This script therefore uses those published per-sensor flags directly,
which also preserves manual "Inv" flagging done during field operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
import pandas as pd

FLAG_CATEGORIES = {
    "warm": {"w"},
    "trh": {"t.min", "t.max", "rh.min", "rh.max"},
    "range": {"low_values", "high_values"},
    "outlier": {"outliersmin", "outliersmax"},
    "inv": {"inv"},
}


PAPER_REFERENCES = {
    "technical_validation": "docs/seneurcity.md:342",
    "flag_description": "docs/seneurcity.md:348",
    "flag_warm": "docs/seneurcity.md:351",
    "flag_trh": "docs/seneurcity.md:352",
    "flag_range": "docs/seneurcity.md:395",
    "flag_outliers": "docs/seneurcity.md:396",
    "flag_inv": "docs/seneurcity.md:411",
    "filter_parameters_table11": "docs/seneurcity.md:379",
    "code_availability": "docs/seneurcity.md:423",
    "filter_function": "docs/seneurcity.md:425",
    "lag_note": "docs/seneurcity.md:421",
}


@dataclass
class StageRecord:
    name: str
    input_rows: int
    removed_rows: int
    output_rows: int
    output_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter SenseurCity data using published QA/QC flags and save "
            "intermediate data at every step."
        )
    )
    parser.add_argument("--data-dir", type=str, default="data/seneurcity")
    parser.add_argument("--pattern", type=str, default="Antwerp_*.csv")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of matching files for quick runs.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["OPCN3PM25", "5325CAT", "5325CST"],
        help=(
            "Sensor columns used for QA/QC row filtering. All three PM2.5 "
            "sensors are retained for downstream experiments."
        ),
    )
    parser.add_argument(
        "--x-cols",
        nargs="+",
        default=[
            "latitude",
            "longitude",
            "SHT31TE",
            "SHT31HE",
            "Absolute_humidity",
            "Td_deficit",
            "BMP280",
        ],
        help="Covariate columns to preserve in saved intermediate outputs.",
    )
    parser.add_argument(
        "--location-cols",
        nargs="+",
        default=["Location.ID", "LocationID", "location_id"],
        help="Candidate location identifier columns to preserve.",
    )
    parser.add_argument(
        "--extra-cols",
        nargs="*",
        default=[],
        help="Any additional columns to preserve.",
    )
    parser.add_argument(
        "--keep-all-cols",
        action="store_true",
        help="If set, read and save all columns (larger and slower).",
    )
    parser.add_argument("--target-col", type=str, default="Ref.PM2.5")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument(
        "--keep-mode",
        type=str,
        choices=["all", "any"],
        default="all",
        help=(
            "all: require every selected sensor to be valid at each stage; "
            "any: keep row if at least one selected sensor remains valid."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/seneurcity_filtered",
        help="Root directory where run subdirectory and artifacts are saved.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional deterministic run name. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--save-compression",
        type=str,
        choices=["none", "gzip"],
        default="none",
        help="Compression for CSV snapshots.",
    )
    return parser.parse_args()


def make_run_dir(output_dir: str, run_name: str | None) -> str:
    if run_name is None:
        run_name = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def discover_input_files(
    data_dir: str, pattern: str, max_files: int | None
) -> list[str]:
    files = sorted(str(p) for p in Path(data_dir).glob(pattern))
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def detect_flag_column_from_columns(
    columns: Sequence[str], sensor_col: str
) -> str | None:
    candidates = [
        f"{sensor_col}_Flag",
        f"{sensor_col}.Flag",
        f"{sensor_col}Flag",
        f"{sensor_col}_flag",
        f"{sensor_col}.flag",
    ]
    for c in candidates:
        if c in columns:
            return c

    normalized_target = f"{sensor_col.lower()}_flag"
    for c in columns:
        normalized = c.lower().replace(".", "_")
        if normalized == normalized_target:
            return c
    return None


def detect_flag_column(df: pd.DataFrame, sensor_col: str) -> str | None:
    return detect_flag_column_from_columns(list(df.columns), sensor_col)


def normalize_flag_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    text = text.strip('"').strip("'")
    parts = [
        p.strip().strip('"').strip("'").lower()
        for p in re.split(r"[,;]", text)
        if p and p.strip()
    ]
    if not parts:
        return ""

    unique_parts = sorted(set(parts))
    return "|".join(unique_parts)


def has_any_flag_token(normalized_flags: pd.Series, tokens: set[str]) -> pd.Series:
    if not tokens:
        return pd.Series(False, index=normalized_flags.index)

    pattern = "|".join(
        [rf"(?:^|\|){re.escape(token.lower())}(?:$|\|)" for token in sorted(tokens)]
    )
    return normalized_flags.str.contains(pattern, regex=True, na=False)


def get_column_as_series(df: pd.DataFrame, col: str) -> pd.Series:
    data = df[col]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return pd.Series(data, index=df.index)


def aggregate_stage_mask(masks: Sequence[pd.Series], keep_mode: str) -> pd.Series:
    if not masks:
        return pd.Series(dtype=bool)

    mat = np.column_stack([m.to_numpy(dtype=bool) for m in masks])
    if keep_mode == "all":
        return pd.Series(np.any(mat, axis=1), index=masks[0].index, dtype=bool)
    return pd.Series(np.all(mat, axis=1), index=masks[0].index, dtype=bool)


def save_snapshot(df: pd.DataFrame, run_dir: str, stem: str, compression: str) -> str:
    use_gzip = compression == "gzip"
    suffix = ".csv.gz" if use_gzip else ".csv"
    out_path = Path(run_dir) / f"{stem}{suffix}"
    if use_gzip:
        df.to_csv(out_path, index=False, compression="gzip")
    else:
        df.to_csv(out_path, index=False)
    return str(out_path)


def apply_filter_stage(
    df: pd.DataFrame,
    invalid_mask: pd.Series,
    run_dir: str,
    stem: str,
    records: list[StageRecord],
    compression: str,
) -> pd.DataFrame:
    invalid_mask = invalid_mask.fillna(False)
    input_rows = len(df)
    out_df = df.loc[~invalid_mask].copy()
    output_rows = len(out_df)
    removed_rows = input_rows - output_rows
    output_file = save_snapshot(out_df, run_dir, stem, compression)
    records.append(
        StageRecord(
            name=stem,
            input_rows=input_rows,
            removed_rows=removed_rows,
            output_rows=output_rows,
            output_file=output_file,
        )
    )
    return out_df


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.output_dir, args.run_name)

    input_files = discover_input_files(args.data_dir, args.pattern, args.max_files)
    print(f"Found {len(input_files)} files")

    first_header_cols = pd.read_csv(input_files[0], nrows=0).columns.tolist()
    preliminary_flag_cols = {
        sensor: detect_flag_column_from_columns(first_header_cols, sensor)
        for sensor in args.sensors
    }
    base_cols = set(args.sensors)
    base_cols.add(args.date_col)
    base_cols.add(args.target_col)
    for col in args.x_cols:
        base_cols.add(col)
    for col in args.location_cols:
        base_cols.add(col)
    for col in args.extra_cols:
        base_cols.add(col)
    for col in preliminary_flag_cols.values():
        if col is not None:
            base_cols.add(col)

    selected_cols: list[str] = sorted(base_cols)

    manifests = []
    frames = []
    for path in input_files:
        header_cols = pd.read_csv(path, nrows=0).columns.tolist()
        p = Path(path)
        manifests.append(
            {
                "path": path,
                "size_bytes": p.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

        if args.keep_all_cols:
            usecols: list[str] | None = None
        else:
            usecols = [c for c in selected_cols if c in header_cols]

        if usecols is None:
            df_i = pd.read_csv(path, low_memory=False)
        else:
            usecols_set = set(usecols)
            df_i = pd.read_csv(
                path,
                usecols=lambda c, cols=usecols_set: bool(c in cols),
                low_memory=False,
            )

        if not args.keep_all_cols:
            for c in selected_cols:
                if c not in df_i.columns:
                    df_i[c] = np.nan
            df_i = df_i[selected_cols]

        ase_id = Path(path).stem.split("_")[-1]
        df_i["ase_id"] = ase_id
        df_i["source_file"] = Path(path).name
        frames.append(df_i)

    df = pd.concat(frames, ignore_index=True)

    records: list[StageRecord] = []
    raw_path = save_snapshot(df, run_dir, "00_raw_concat", args.save_compression)
    records.append(
        StageRecord(
            name="00_raw_concat",
            input_rows=len(df),
            removed_rows=0,
            output_rows=len(df),
            output_file=raw_path,
        )
    )

    df[args.date_col] = pd.to_datetime(
        get_column_as_series(df, args.date_col),
        utc=True,
        errors="coerce",
    )
    invalid_date_mask = get_column_as_series(df, args.date_col).isna()
    df = apply_filter_stage(
        df=df,
        invalid_mask=invalid_date_mask,
        run_dir=run_dir,
        stem="01_after_valid_date",
        records=records,
        compression=args.save_compression,
    )

    if "location_id" not in df.columns:
        if "LocationID" in df.columns:
            df["location_id"] = df["LocationID"]
        else:
            df["location_id"] = ""

    flag_col_map: dict[str, str | None] = {}
    for sensor in args.sensors:
        flag_col_map[sensor] = detect_flag_column(df, sensor)
        if flag_col_map[sensor] is None:
            print(
                f"Warning: no flag column found for sensor {sensor}; "
                "treating flags as empty for that sensor."
            )

    for sensor in args.sensors:
        fcol = flag_col_map[sensor]
        norm_col = f"flag_norm__{sensor}"
        if fcol is None:
            df[norm_col] = ""
        else:
            df[norm_col] = get_column_as_series(df, fcol).map(normalize_flag_text)

        df[f"mask__{sensor}__warm"] = has_any_flag_token(
            get_column_as_series(df, norm_col), FLAG_CATEGORIES["warm"]
        )
        df[f"mask__{sensor}__trh"] = has_any_flag_token(
            get_column_as_series(df, norm_col), FLAG_CATEGORIES["trh"]
        )
        df[f"mask__{sensor}__range"] = has_any_flag_token(
            get_column_as_series(df, norm_col), FLAG_CATEGORIES["range"]
        )
        df[f"mask__{sensor}__outlier"] = has_any_flag_token(
            get_column_as_series(df, norm_col), FLAG_CATEGORIES["outlier"]
        )
        df[f"mask__{sensor}__inv"] = has_any_flag_token(
            get_column_as_series(df, norm_col), FLAG_CATEGORIES["inv"]
        )
        df[f"mask__{sensor}__missing_value"] = get_column_as_series(df, sensor).isna()

    with_flags_path = save_snapshot(
        df, run_dir, "02_with_flag_masks", args.save_compression
    )
    records.append(
        StageRecord(
            name="02_with_flag_masks",
            input_rows=len(df),
            removed_rows=0,
            output_rows=len(df),
            output_file=with_flags_path,
        )
    )

    df_with_flags = df.copy()

    warm_masks = [
        get_column_as_series(df, f"mask__{sensor}__warm") for sensor in args.sensors
    ]
    row_mask_warm = aggregate_stage_mask(warm_masks, args.keep_mode)
    df = apply_filter_stage(
        df=df,
        invalid_mask=row_mask_warm,
        run_dir=run_dir,
        stem="03_after_warming_filter",
        records=records,
        compression=args.save_compression,
    )

    trh_masks = [
        get_column_as_series(df, f"mask__{sensor}__trh") for sensor in args.sensors
    ]
    row_mask_trh = aggregate_stage_mask(trh_masks, args.keep_mode)
    df = apply_filter_stage(
        df=df,
        invalid_mask=row_mask_trh,
        run_dir=run_dir,
        stem="04_after_trh_filter",
        records=records,
        compression=args.save_compression,
    )

    outlier_masks = [
        (
            get_column_as_series(df, f"mask__{sensor}__range")
            | get_column_as_series(df, f"mask__{sensor}__outlier")
        )
        for sensor in args.sensors
    ]
    row_mask_outlier = aggregate_stage_mask(outlier_masks, args.keep_mode)
    df = apply_filter_stage(
        df=df,
        invalid_mask=row_mask_outlier,
        run_dir=run_dir,
        stem="05_after_outlier_filter",
        records=records,
        compression=args.save_compression,
    )

    inv_masks = [
        get_column_as_series(df, f"mask__{sensor}__inv") for sensor in args.sensors
    ]
    row_mask_inv = aggregate_stage_mask(inv_masks, args.keep_mode)
    df = apply_filter_stage(
        df=df,
        invalid_mask=row_mask_inv,
        run_dir=run_dir,
        stem="06_after_inv_filter",
        records=records,
        compression=args.save_compression,
    )

    row_mask_missing_required = get_column_as_series(df, args.target_col).isna()
    for sensor in args.sensors:
        row_mask_missing_required = (
            row_mask_missing_required | get_column_as_series(df, sensor).isna()
        )

    df = apply_filter_stage(
        df=df,
        invalid_mask=row_mask_missing_required,
        run_dir=run_dir,
        stem="07_after_required_value_filter",
        records=records,
        compression=args.save_compression,
    )

    df = df.sort_values(by=[args.date_col, "source_file"], kind="mergesort")
    final_path = save_snapshot(
        df,
        run_dir,
        "08_final_filtered_sorted",
        args.save_compression,
    )
    records.append(
        StageRecord(
            name="08_final_filtered_sorted",
            input_rows=len(df),
            removed_rows=0,
            output_rows=len(df),
            output_file=final_path,
        )
    )

    reason_counts: dict[str, dict[str, int]] = {}
    for sensor in args.sensors:
        reason_counts[sensor] = {
            "warm": (
                int(df_with_flags[f"mask__{sensor}__warm"].sum())
                if f"mask__{sensor}__warm" in df_with_flags.columns
                else 0
            ),
            "trh": (
                int(df_with_flags[f"mask__{sensor}__trh"].sum())
                if f"mask__{sensor}__trh" in df_with_flags.columns
                else 0
            ),
            "range": (
                int(df_with_flags[f"mask__{sensor}__range"].sum())
                if f"mask__{sensor}__range" in df_with_flags.columns
                else 0
            ),
            "outlier": (
                int(df_with_flags[f"mask__{sensor}__outlier"].sum())
                if f"mask__{sensor}__outlier" in df_with_flags.columns
                else 0
            ),
            "inv": (
                int(df_with_flags[f"mask__{sensor}__inv"].sum())
                if f"mask__{sensor}__inv" in df_with_flags.columns
                else 0
            ),
        }

    summary = {
        "run_dir": run_dir,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "data_dir": args.data_dir,
            "pattern": args.pattern,
            "file_count": len(input_files),
            "files": manifests,
        },
        "config": {
            "sensors": args.sensors,
            "target_col": args.target_col,
            "date_col": args.date_col,
            "keep_mode": args.keep_mode,
            "save_compression": args.save_compression,
        },
        "flag_columns": flag_col_map,
        "pipeline_records": [record.__dict__ for record in records],
        "reason_counts_on_full_flagged_df": reason_counts,
        "paper_references": PAPER_REFERENCES,
    }

    summary_path = Path(run_dir) / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Filtering complete.")
    print(f"Run directory: {run_dir}")
    print(f"Final filtered file: {final_path}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
