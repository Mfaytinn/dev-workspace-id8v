# Neural Conjugate Aggregation

Reproduction scripts for Tables 1–6 and the
anchored/unanchored non-identifiability figure.

## Install

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). From this directory:

```sh
git clone --recursive <this repository>
uv sync --locked
```

After a plain clone, restore the submodules with
`git submodule update --init`. The numerical dependencies are pinned. The
upstream VELI source and license are pinned as a submodule at
`third_party/veli` ([YahiDar/Veli](https://github.com/YahiDar/Veli), commit
`83e2804`) and are required to run the experiments. The JRC R toolbox is
pinned at `third_party/airsenseur-calibration`
([ec-jrc/airsenseur-calibration](https://github.com/ec-jrc/airsenseur-calibration),
commit `a6cabfa`) and is only needed to clean raw SensEURCity data.

## Supply prepared data

Place inputs at `data/seneurcity_preprocessed/Antwerp_<station-id>.csv` and
`data/cairsense_preprocessed.xlsx`. To produce them from raw data, use the
scripts in `third_party/preprocess/`:

- **SensEURCity:** `bash third_party/preprocess/seneurcity.sh` (requires R and
  the `third_party/airsenseur-calibration` submodule).
- **CAIRSENSE:** `uv run --locked python third_party/preprocess/cairsense_preprocess.py`.

Expected column schemas are defined in `configs/`. The Toy Spatial dataset is
generated automatically when missing.

## Run in order

```sh
uv run --locked python -m paper_reproduce.run_temporal_calendar
uv run --locked python -m paper_reproduce.table1_predictivePerformance
uv run --locked python -m paper_reproduce.table2_conformalMetrics
uv run --locked python -m paper_reproduce.table3_biasAblation
uv run --locked python -m paper_reproduce.table4_hyperparameterSensitivity
uv run --locked python -m paper_reproduce.table5_dataEfficiency
uv run --locked python -m paper_reproduce.table6_anchorSensitivity
uv run --locked python -m paper_reproduce.plot_non_identifiability
```

The calendar script trains the shared full-sensor real-data reference models
and baselines. Tables use seeds 0, 1, and 42; the figure defaults to seed 42.
Calendar windows are 56/7/7/7 days for training/validation/calibration/test,
with 36-hour gaps and at least 80% hourly completeness. Nine SensEURCity folds
and two CAIRSENSE folds are evaluated. Real models train for
400 epochs and select weights by sensor validation NLL. Toy uses random splits.

All probabilistic point predictions are raw-space posterior medians. Table 2
excludes a held-out sensor only from calibration/test fusion, keeping the trained
anchor and all three sensors during training. Calibration settings are fixed in
`configs/tables/conformal.yaml`. The explicit raw-sensor variant is:

```sh
uv run --locked python -m paper_reproduce.table2_conformalMetrics --calibration-target raw_sensor
```

Use `--help` for published datasets/seeds, output paths, and figure controls.
For Toy alone, skip the calendar command and add
`--datasets toy_spatial --seeds 0` to the remaining commands. CPU is supported;
independent training jobs use one process per visible CUDA device when available.

## Outputs

Outputs go to `results/tables/<script-name>/`: measurement
and summary CSVs, plus PDF/PNG figures. Calendar and Table 1 model files contain
only weights and their resolved configuration/preprocessing. The figure
refits both models and writes validation trajectories; `--plot-only` replots them.

## Reference results and validation

`reference_results/` contains measured `results.csv` and `table.csv` for each
table, plus Table 4's `grid.csv`.
