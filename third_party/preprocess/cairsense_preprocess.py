"""Merge the CAIRSENSE AirAssure and Wind sheets into a single workbook."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_RAW = Path("data/cairsense_raw.xlsx")
DEFAULT_OUT = Path("data/cairsense_preprocessed.xlsx")

KEEP_AIR = ["date", "AirAssure1", "AirAssure2", "AirAssure3", "TEMP", "RHAMB", "SoC"]
KEEP_WIND = ["date", "WS", "WD"]


def merge_cairsense(
    raw: Path, out: Path, keep_missing_reference: bool = False
) -> pd.DataFrame:
    air = pd.read_excel(raw, sheet_name="AirAssure").rename(
        columns={"timestamp": "date"}
    )
    wind = pd.read_excel(raw, sheet_name="Wind").rename(columns={"Date": "date"})
    air["date"] = pd.to_datetime(air["date"], errors="coerce")
    wind["date"] = pd.to_datetime(wind["date"], errors="coerce")
    required = KEEP_AIR + KEEP_WIND[1:]
    if keep_missing_reference:
        required = [col for col in required if col != "SoC"]
    merged = (
        air[KEEP_AIR]
        .merge(wind[KEEP_WIND], on="date", how="left")
        .dropna(subset=required)
        .sort_values("date")
        .reset_index(drop=True)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out, index=False)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=DEFAULT_RAW, help="Source workbook."
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Merged output workbook."
    )
    parser.add_argument(
        "--keep-missing-reference",
        action="store_true",
        help="Retain sensor-complete rows even when SoC is unavailable.",
    )
    args = parser.parse_args()

    if args.out.exists():
        print(f"[INFO] Using existing merged file: {args.out}")
        return 0
    if not args.raw.exists():
        print(f"[ERROR] Missing {args.raw}", file=sys.stderr)
        return 1

    print(f"[INFO] Building merged CAIRSENSE file: {args.out}")
    merged = merge_cairsense(args.raw, args.out, args.keep_missing_reference)
    print(f"wrote {args.out} shape {merged.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
