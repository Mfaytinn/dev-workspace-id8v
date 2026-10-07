"""Past-only calendar folds and training-only preprocessing."""

from pathlib import Path

import pandas as pd
import yaml

from ncam.experiments.calendar_folds import calendar_fold
from ncam.experiments.config import configured_anchor
from ncam.experiments.preparation import load_dataset, prepare_fold, prepare_toy_fold
from ncam.experiments.records import SplitFrames
from ncam.experiments.splits import reserve_temporal_holdout

DEFAULT_CALENDAR_CONFIG = Path("configs/tables/calendar.yaml")
WINDOW_KEYS = ("train_days", "val_days", "cal_days", "test_days", "gap_hours")


def calendar_spec(config: dict | str | Path | None = None) -> dict:
    """Load the shared calendar specification from a run config or path."""
    if isinstance(config, dict):
        config = config.get("calendar_config")
    path = DEFAULT_CALENDAR_CONFIG if config is None else Path(config)
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid calendar specification: {path}")
    missing = set(WINDOW_KEYS) - spec.keys()
    if missing:
        raise ValueError(f"Calendar specification is missing {sorted(missing)}")
    return spec


def apply_calendar_spec(cfg: dict, spec: dict | None = None) -> dict:
    """Apply shared calendar and conformal settings before caller overrides."""
    spec = calendar_spec(cfg) if spec is None else spec
    resolved = dict(cfg)
    resolved.update(
        calendar_config=str(cfg.get("calendar_config", DEFAULT_CALENDAR_CONFIG)),
        alpha=spec["alpha"],
        conformal_scheme=spec["conformal_scheme"],
        conformal_block_size=spec["conformal_block_size"],
        conformal_score_space=spec["conformal_score_space"],
        epochs=spec["epochs"],
        min_observed_fraction=spec.get("min_observed_fraction", 0.8),
        point_prediction_summary=spec.get("point_prediction_summary", "median"),
    )
    resolved.update(spec.get("model_overrides", {}))
    if cfg.get("model") == "unanchored":
        # The identifiability ablation inherits resolved model settings rather
        # than automatically selecting a different candidate from the main bank.
        resolved.setdefault("model_candidates", {})
    dataset_candidates = (
        spec.get("datasets", {}).get(cfg.get("dataset"), {}).get("model_candidates")
    )
    if dataset_candidates is not None and "model_candidates" not in resolved:
        resolved["model_candidates"] = dataset_candidates
    resolved["calendar_spec"] = spec
    return resolved


def calendar_window_options(spec: dict) -> dict:
    return {
        **{key: spec[key] for key in WINDOW_KEYS},
        "min_observed_fraction": spec.get("min_observed_fraction", 0.8),
    }


def _reserved_holdout(frame: pd.DataFrame, cfg: dict, spec: dict):
    if not spec.get("exclude_reserved_holdout", False):
        return None
    _, holdout = reserve_temporal_holdout(
        frame,
        block_size_days=int(cfg["block_size_days"]),
        gap_hours=int(spec["gap_hours"]),
        cal_frac=float(cfg["holdout_cal_frac"]),
        test_frac=float(cfg["holdout_test_frac"]),
    )
    return holdout


def calendar_plan(
    frame: pd.DataFrame, cfg: dict, *, spec: dict | None = None
) -> list[tuple[str, SplitFrames, dict]]:
    """Build fixed calendar folds using only their configured endpoints."""
    spec = calendar_spec(cfg) if spec is None else spec
    dataset = cfg["dataset"]
    if dataset not in spec["datasets"]:
        raise ValueError(f"No calendar endpoints configured for {dataset}")
    reserved = _reserved_holdout(frame, cfg, spec)
    reserved_start = (
        pd.Timestamp(reserved["splits"]["cal"]["first_timestamp"])
        if reserved is not None
        else None
    )
    plan = []
    seen = set()
    for value in spec["datasets"][dataset]["test_ends"]:
        endpoint = pd.Timestamp(value)
        if reserved_start is not None and endpoint > reserved_start:
            raise ValueError("Calendar test window enters the reserved holdout")
        frames, windows = calendar_fold(
            frame,
            endpoint,
            **calendar_window_options(spec),
            min_train_rows=int(spec.get("min_train_rows", 600)),
            min_val_rows=int(spec.get("min_val_rows", 120)),
            min_cal_timestamps=int(spec.get("min_cal_timestamps", 119)),
            min_test_timestamps=int(spec.get("min_test_timestamps", 24)),
            reference_col=cfg.get("y_col"),
        )
        fold_id = endpoint.strftime("%Y%m%dT%H")
        if fold_id in seen:
            raise ValueError(f"Duplicate calendar endpoint: {endpoint}")
        seen.add(fold_id)
        plan.append(
            (
                fold_id,
                frames,
                {
                    "test_end": endpoint.isoformat(),
                    "windows": windows,
                    "reserved_holdout": reserved,
                },
            )
        )
    return plan


def _labeled_frames(frames: SplitFrames, y_col: str) -> SplitFrames:
    return frames._replace(
        cal=frames.cal.dropna(subset=[y_col]).copy(),
        test=frames.test.dropna(subset=[y_col]).copy(),
    )


def prepare_calendar_fold(frames, cfg, fold_id, device, *, metadata, fitted=False):
    cfg = dict(
        cfg,
        calendar_fold_id=fold_id,
        calendar_test_end=metadata["test_end"],
        calendar_windows=metadata["windows"],
    )
    fold = prepare_fold(
        _labeled_frames(frames, cfg["y_col"]),
        cfg,
        device,
        subsample=True,
        fitted=fitted,
    )
    if cfg["model"] == "anchored":
        anchor, index = configured_anchor(cfg["b_cols"], cfg["anchor_sensor"])
        fold.config.update(anchor_sensor=anchor, anchor_idx=index)
    return fold


def reconstruct_fold(cfg, device):
    """Recreate the reference split from its resolved model configuration."""
    frame, loaded = load_dataset(cfg["dataset"], cfg, saved_config=cfg)
    if cfg["dataset"] == "toy_spatial":
        return prepare_toy_fold(frame, loaded, device, fitted=True)
    spec = cfg["calendar_spec"]
    options = calendar_window_options(spec)
    frames, windows = calendar_fold(
        frame,
        pd.Timestamp(cfg["calendar_test_end"]),
        **options,
        min_train_rows=int(spec.get("min_train_rows", 600)),
        min_val_rows=int(spec.get("min_val_rows", 120)),
        min_cal_timestamps=int(spec.get("min_cal_timestamps", 119)),
        min_test_timestamps=int(spec.get("min_test_timestamps", 24)),
        reference_col=cfg["y_col"],
    )
    return prepare_calendar_fold(
        frames,
        loaded,
        cfg["calendar_fold_id"],
        device,
        metadata={"test_end": cfg["calendar_test_end"], "windows": windows},
        fitted=True,
    )
