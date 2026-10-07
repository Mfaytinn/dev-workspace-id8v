#!/usr/bin/env bash
# Full SensEURCity preprocessing pipeline: raw per-station CSVs under
# data/seneurcity_raw/ → flag-cleaned → lag-corrected → QA/QC-filtered
# dataset under data/seneurcity_preprocessed/, which is what
# configs/seneurcity.yaml (data_dir) points to. Idempotent: re-run skips
# stages whose output dir already contains CSVs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RAW_DIR="data/seneurcity_raw"
FLAGS_DIR="data/seneurcity_flags_cleaned"
LAG_DIR="data/seneurcity_lag_fixed"
PREPROCESSED_DIR="data/seneurcity_preprocessed"

if ! compgen -G "$FLAGS_DIR/*.csv" > /dev/null; then
  Rscript third_party/preprocess/seneurcity_clean_flags.R \
    --in-dir "$RAW_DIR" \
    --out-dir "$FLAGS_DIR"
fi

if ! compgen -G "$LAG_DIR/*.csv" > /dev/null; then
  Rscript third_party/preprocess/seneurcity_fix_lag.R \
    --in-dir "$FLAGS_DIR" \
    --out-dir "$LAG_DIR"
fi

if ! compgen -G "$PREPROCESSED_DIR/*.csv" > /dev/null; then
  uv run --locked python third_party/preprocess/seneurcity_filter.py \
    --data-dir "$LAG_DIR" \
    --output-dir "$PREPROCESSED_DIR" \
    --run-name preprocessed
fi
