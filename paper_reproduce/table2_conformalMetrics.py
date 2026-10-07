"""Table 2: full-sensor posterior and adaptive raw held-out-sensor intervals.

The default --raw-scale posterior normalizes raw residuals by the half raw
quantile span of the remaining-sensor latent posterior at Phi(-1), Phi(1).
Use --raw-scale constant to reproduce earlier absolute raw scores, or sensor
for the held-out sensor predictive-spread comparison. The historical corrected
surrogate needs --calibration-target bias_corrected_sensor --raw-scale constant.
"""

import json
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
import yaml

from ncam.calibration import conformal_order_statistic
from ncam.checkpoints import load_model
from ncam.evaluation import (
    _log_floor_for_targets,
    _transform_targets_to_model_space,
    posterior_point_prediction,
)
from ncam.experiments.calendar_protocol import reconstruct_fold
from ncam.experiments.preparation import DATASET_SENSORS, load_dataset, prepare_toy_fold
from ncam.experiments.splits import build_random_split_indices
from ncam.experiments.training import fit_fold
from ncam.models import _predict_posterior_from_params, _variance_from_log
from ncam.temporal import (
    TemporalCPSettings,
    calibration_window,
    temporal_bounds,
    temporal_threshold,
    timestamp_coverage,
)
from ncam.training import initialize_runtime
from paper_reproduce._common import (
    base_config,
    configure_calendar_parser,
    ensure_data,
    parse_args,
)
from paper_reproduce._reporting import summarize_results

ORACLE = "Oracle Split Conformal"
HS_CP = "Held-Out Sensor CP"
RAW_SENSOR_CP = "Held-Out Raw Sensor CP"
ADAPTIVE_RAW_SENSOR_CP = "Adaptive Held-Out Raw Sensor CP"
RAW_SCALES = ("constant", "posterior", "sensor")
POSTERIOR_PREDICTIVE = "Fused Posterior Predictive"
DATASET_LABELS = {
    "toy_spatial": "Toy",
    "seneurcity": "SensEURCity",
    "cairsense": "CAIRSENSE",
}
DEFAULT_TEMPORAL_PROFILE = Path("configs/tables/conformal.yaml")


def configure_parser(parser):
    configure_calendar_parser(parser)
    parser.set_defaults(alpha=0.2)
    parser.add_argument(
        "--calibration-target",
        choices=("bias_corrected_sensor", "raw_sensor"),
        default="raw_sensor",
        help="Table 2 uses raw sensor residuals around posterior medians",
    )
    parser.add_argument(
        "--raw-scale",
        choices=RAW_SCALES,
        default="posterior",
        help=(
            "Raw-score scale: remaining-sensor posterior spread (default), "
            "held-out sensor predictive spread, or constant for earlier absolute scores"
        ),
    )


def temporal_settings(dataset, held_out=None):
    selected = yaml.safe_load(DEFAULT_TEMPORAL_PROFILE.read_text())["datasets"][dataset]
    if held_out:
        selected = {
            **selected,
            **selected.get("heldout", {}),
            **selected.get("sensor_overrides", {}).get(held_out, {}),
        }
    return TemporalCPSettings(
        block_size=selected["block_size"],
        window_timestamps=selected["window_timestamps"],
        scheme=selected["scheme"],
        scale_power=1.0,
    )


def posterior_predictive_result(mean, std, cfg, cal_frame, test_frame, *, temporal):
    """Evaluate Gaussian predictive bounds without fitting a calibration threshold."""
    alpha = float(cfg["alpha"])
    threshold = NormalDist().inv_cdf(1 - alpha / 2)
    lower, upper = temporal_bounds(
        mean,
        std,
        threshold,
        TemporalCPSettings(block_size=1, scale_power=1.0),
        obs_transform=cfg["obs_transform"],
        target_floor=(
            _log_floor_for_targets(cfg) if cfg["obs_transform"] == "log" else None
        ),
    )
    if len(lower) != len(test_frame) or not (
        np.isfinite(lower).all() and np.isfinite(upper).all()
    ):
        raise ValueError("Invalid posterior predictive bounds")
    target = test_frame[cfg["y_col"]].to_numpy(dtype=np.float32)
    hit = (lower <= target) & (target <= upper)
    coverage = float(hit.mean())
    joint = timestamp_coverage(hit, test_frame.date.to_numpy()) if temporal else np.nan
    return {
        "threshold": threshold,
        "coverage_primary": joint if temporal else coverage,
        "coverage_reference_row": coverage,
        "coverage_sensor_row": np.nan,
        "coverage_reference_timestamp": joint,
        "coverage_sensor_timestamp": np.nan,
        "width": float(np.maximum(upper - lower, 0).mean()),
        "cal_rows": len(cal_frame),
        "cal_timestamps": cal_frame.date.nunique() if temporal else np.nan,
        "cal_rows_used": 0,
        "cal_timestamps_used": 0,
        "test_rows": len(test_frame),
        "test_timestamps": test_frame.date.nunique() if temporal else np.nan,
        "alpha": alpha,
        "score_space": "none",
    }


def raw_predictive_spread(mean, std, obs_transform):
    """Half the raw quantile span at Gaussian probabilities Phi(-1), Phi(1).

    This positive scale uses transformed quantiles, never a raw posterior mean.
    The fixed 1e-8 floor is in raw measurement units.
    """
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    if mean.shape != std.shape or not (
        np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    ):
        raise ValueError("Invalid predictive location or scale")
    if obs_transform == "none":
        spread = std
    elif obs_transform in {"log", "log1p"}:
        # expm1 has the same quantile difference as exp. This expression
        # avoids subtracting nearly equal transformed quantiles.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            spread = np.exp(mean + std) * (-np.expm1(-2 * std)) / 2
    else:
        raise ValueError(f"Unsupported observation transform: {obs_transform}")
    if not np.isfinite(spread).all():
        raise ValueError("Nonfinite raw predictive spread")
    return np.maximum(spread, 1e-8)


@torch.no_grad()
def sensor_role_predictions(
    model,
    x,
    sensors,
    *,
    held_out_index=None,
    sigma_y=0.0,
    correct_surrogate=True,
    raw_scale="constant",
    obs_transform="none",
):
    """Keep the learned anchor/heads; slice fusion terms, never retrain or reanchor."""
    mu0, log_var0, log_var_j, gains, offsets = model(x)
    indices = [j for j in range(sensors.shape[1]) if j != held_out_index]
    if held_out_index is not None and held_out_index not in range(sensors.shape[1]):
        raise ValueError("Held-out sensor index is out of range")
    mean, variance = _predict_posterior_from_params(
        sensors[:, indices],
        mu0,
        log_var0,
        log_var_j[:, indices],
        gains[:, indices],
        offsets[:, indices],
        variance_transform=model.variance_transform,
    )
    result = {
        "mean": mean.ravel().cpu().numpy(),
        "std": (variance.ravel().clamp_min(0) + float(sigma_y) ** 2)
        .clamp_min(1e-8)
        .sqrt()
        .cpu()
        .numpy(),
    }
    if raw_scale not in RAW_SCALES:
        raise ValueError("Unknown raw sensor scale")
    if held_out_index is not None and raw_scale != "constant":
        scale_mean, scale_variance = mean.ravel(), variance.ravel().clamp_min(0)
        if raw_scale == "sensor":
            gain, offset = gains[:, held_out_index], offsets[:, held_out_index]
            sensor_variance = _variance_from_log(
                log_var_j[:, held_out_index], method=model.variance_transform
            )
            scale_mean = gain * scale_mean + offset
            scale_variance = gain.square() * scale_variance + sensor_variance
        result["raw_scale"] = raw_predictive_spread(
            scale_mean.cpu().numpy(),
            scale_variance.clamp_min(1e-8).sqrt().cpu().numpy(),
            obs_transform,
        )
    if held_out_index is not None and correct_surrogate:
        gain = gains[:, held_out_index]
        offset = offsets[:, held_out_index]
        if not torch.isfinite(gain).all() or (gain.abs() < 1e-8).any():
            raise ValueError("Held-out sensor gains cannot be inverted")
        result.update(
            surrogate=((sensors[:, held_out_index] - offset) / gain).cpu().numpy(),
            gain=gain.cpu().numpy(),
            offset=offset.cpu().numpy(),
        )
    if any(not np.isfinite(values).all() for values in result.values()):
        raise ValueError("Invalid posterior or held-out sensor surrogate")
    return result


def evaluate_role(
    cal, test, cal_frame, test_frame, cfg, held_out, settings=None, *, row_details=None
):
    """Fit thresholds solely to calibration targets; evaluate identical test bounds."""
    alpha = float(cfg["alpha"])
    reference = cfg["y_col"]
    raw = bool(held_out) and cfg.get("heldout_calibration_target") == "raw_sensor"
    raw_scale = cfg.get("heldout_raw_scale", "constant") if raw else "constant"
    if raw_scale not in RAW_SCALES:
        raise ValueError("Unknown raw sensor scale")
    adaptive = raw and raw_scale != "constant"
    if raw:
        target = cal_frame[held_out].to_numpy(dtype=np.float32)
        converted = []
        for prediction in (cal, test):
            mean = torch.as_tensor(prediction["mean"])
            median = posterior_point_prediction(
                mean, torch.zeros_like(mean), cfg["obs_transform"], "median"
            ).numpy()
            scale = (
                np.asarray(prediction["raw_scale"], dtype=np.float64).ravel()
                if adaptive
                else np.ones_like(median)
            )
            if scale.shape != median.shape or not (
                np.isfinite(scale).all() and (scale > 0).all()
            ):
                raise ValueError("Raw sensor scales must be positive and match rows")
            converted.append(dict(prediction, mean=median, std=scale))
        cal, test = converted
    elif held_out:
        target = cal["surrogate"]
    else:
        target = _transform_targets_to_model_space(
            torch.as_tensor(cal_frame[reference].to_numpy(dtype=np.float32)), cfg
        ).numpy()
    temporal = settings is not None
    settings = settings or TemporalCPSettings(block_size=1, scheme="cyclic")
    if raw:
        settings = replace(settings, scale_power=1.0 if adaptive else 0.0)
    if not raw and settings.scale_power != 1:
        raise ValueError("Section 3.6 requires normalized model-space scores")
    model_lower = np.empty(
        len(test_frame), dtype=np.float64 if adaptive else np.float32
    )
    model_upper = np.empty_like(model_lower)
    lower, upper = np.empty_like(model_lower), np.empty_like(model_lower)
    used = np.zeros(len(cal_frame), dtype=bool)
    thresholds = []
    locations = np.unique(test_frame.location_id) if temporal else [None]
    for location in locations:
        cal_mask = (
            cal_frame.location_id.to_numpy() == location
            if temporal
            else np.ones(len(cal_frame), dtype=bool)
        )
        test_mask = (
            test_frame.location_id.to_numpy() == location
            if temporal
            else np.ones(len(test_frame), dtype=bool)
        )
        if not cal_mask.any():
            raise ValueError(f"No calibration rows for location {location}")
        if temporal:
            times = cal_frame.date.to_numpy()[cal_mask]
            selected = calibration_window(times, settings)
            used[np.flatnonzero(cal_mask)[selected]] = True
            threshold = temporal_threshold(
                cal["mean"][cal_mask],
                cal["std"][cal_mask],
                target[cal_mask],
                times,
                settings,
                alpha=alpha,
            )
        else:
            used[:] = True
            threshold = conformal_order_statistic(
                np.abs(target - cal["mean"]) / cal["std"], alpha
            )
        if not np.isfinite(threshold):
            raise ValueError("Insufficient calibration rows for a finite threshold")
        mean, std = test["mean"][test_mask], test["std"][test_mask]
        model_lower[test_mask] = mean - threshold * std
        model_upper[test_mask] = mean + threshold * std
        lower[test_mask], upper[test_mask] = temporal_bounds(
            mean,
            std,
            threshold,
            settings,
            obs_transform="none" if raw else cfg["obs_transform"],
            target_floor=(
                _log_floor_for_targets(cfg)
                if not raw and cfg["obs_transform"] == "log"
                else None
            ),
            lower_support=0.0 if temporal else None,
        )
        thresholds.append(threshold)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Nonfinite raw-space interval bounds")
    ref = test_frame[reference].to_numpy(dtype=np.float32)
    reference_hit = (lower <= ref) & (ref <= upper)
    raw_sensor = test_frame[held_out].to_numpy(dtype=np.float32) if held_out else None
    raw_hit = (lower <= raw_sensor) & (raw_sensor <= upper) if held_out else None
    if raw and row_details is not None:
        columns = list(
            dict.fromkeys(
                column
                for column in ["date", "location_id", *cfg.get("x_cols", [])]
                if column in test_frame
            )
        )
        details = test_frame[columns].reset_index(drop=True).copy()
        retained = [sensor for sensor in cfg["b_cols"] if sensor != held_out]
        details["retained_sensor_disagreement"] = (
            test_frame[retained].std(axis=1, ddof=0).to_numpy()
        )
        details["held_out_sensor"] = held_out
        details["median"] = test["mean"]
        details["raw_scale"] = test["std"]
        details["lower"], details["upper"] = lower, upper
        details["width"] = np.maximum(upper - lower, 0)
        details["sensor_hit"], details["reference_hit"] = raw_hit, reference_hit
        row_details.append(details)
    sensor_hit = (
        raw_hit
        if raw
        else (
            (model_lower <= test["surrogate"]) & (test["surrogate"] <= model_upper)
            if held_out
            else None
        )
    )
    return {
        "threshold": float(np.mean(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "coverage_primary": float((sensor_hit if held_out else reference_hit).mean()),
        "coverage_reference_row": float(reference_hit.mean()),
        "coverage_sensor_row": float(sensor_hit.mean()) if held_out else np.nan,
        "coverage_raw_sensor_row": float(raw_hit.mean()) if held_out else np.nan,
        "coverage_reference_timestamp": (
            timestamp_coverage(reference_hit, test_frame.date) if temporal else np.nan
        ),
        "coverage_sensor_timestamp": (
            timestamp_coverage(sensor_hit, test_frame.date)
            if temporal and held_out
            else np.nan
        ),
        "width": float(np.maximum(upper - lower, 0).mean(dtype=np.float64)),
        "cal_rows": len(cal_frame),
        "cal_rows_used": int(used.sum()),
        "cal_timestamps": cal_frame.date.nunique() if temporal else np.nan,
        "cal_timestamps_used": cal_frame.date[used].nunique() if temporal else np.nan,
        "test_rows": len(test_frame),
        "test_timestamps": test_frame.date.nunique() if temporal else np.nan,
        "alpha": alpha,
        "score_space": (
            "raw_normalized"
            if adaptive
            else "raw_absolute" if raw else "model_normalized"
        ),
        "raw_scale": raw_scale if raw else "not_applicable",
        "coverage_unit": "per_location" if temporal else "row",
        "calibration_target": (
            "raw_sensor"
            if raw
            else "bias_corrected_sensor" if held_out else "reference"
        ),
        **settings.to_dict(),
    }


def _predictions(model, split, cfg, *, correct_surrogate=True):
    return {
        held_out
        or "reference": sensor_role_predictions(
            model,
            split.x,
            split.sensors,
            held_out_index=(cfg["b_cols"].index(held_out) if held_out else None),
            sigma_y=cfg["sigma_y"],
            correct_surrogate=correct_surrogate,
            raw_scale=cfg.get("heldout_raw_scale", "constant"),
            obs_transform=cfg["obs_transform"],
        )
        for held_out in (None, *cfg["b_cols"])
    }


def _partition_rows(
    cal,
    test,
    cal_frame,
    test_frame,
    cfg,
    common,
    settings=None,
    *,
    role_settings=None,
    row_details=None,
):
    rows = []
    for held_out in (None, *cfg["b_cols"]):
        key = held_out or "reference"
        selected_settings = (role_settings or {}).get(held_out, settings)
        result = evaluate_role(
            cal[key],
            test[key],
            cal_frame,
            test_frame,
            cfg,
            held_out,
            selected_settings,
            row_details=row_details,
        )
        rows.append(
            dict(
                common,
                **result,
                method=(
                    (
                        ADAPTIVE_RAW_SENSOR_CP
                        if cfg.get("heldout_raw_scale", "constant") != "constant"
                        else RAW_SENSOR_CP
                    )
                    if held_out
                    and cfg.get("heldout_calibration_target") == "raw_sensor"
                    else HS_CP if held_out else ORACLE
                ),
                held_out_sensor=held_out or "",
                fusion_sensors=json.dumps([s for s in cfg["b_cols"] if s != held_out]),
            )
        )
    predictive = posterior_predictive_result(
        test["reference"]["mean"],
        test["reference"]["std"],
        cfg,
        cal_frame,
        test_frame,
        temporal=settings is not None,
    )
    rows.append(
        dict(
            common,
            **predictive,
            coverage_raw_sensor_row=np.nan,
            method=POSTERIOR_PREDICTIVE,
            held_out_sensor="",
            fusion_sensors=json.dumps(cfg["b_cols"]),
        )
    )
    return rows


def calendar_rows(args, dataset, seed):
    paths = sorted(args.calendar_study.glob(f"models/{dataset}/*/seed_{seed}/model.pt"))
    if not paths:
        raise FileNotFoundError("Run paper_reproduce.run_temporal_calendar first")
    device, rows = initialize_runtime(seed), []
    settings = temporal_settings(dataset)
    for path in paths:
        model, cfg = load_model(path, device)
        fold = reconstruct_fold(cfg, device)
        raw = args.calibration_target == "raw_sensor"
        cfg["heldout_raw_scale"] = args.raw_scale
        cal = _predictions(model, fold.cal, cfg, correct_surrogate=not raw)
        test = _predictions(model, fold.test, cfg, correct_surrogate=not raw)
        common = {
            "dataset": dataset,
            "seed": seed,
            "origin": cfg["calendar_fold_id"],
            "split": np.nan,
            "anchor_sensor": cfg["anchor_sensor"],
        }
        rows.extend(
            _partition_rows(
                cal,
                test,
                fold.frames.cal,
                fold.frames.test,
                dict(
                    cfg,
                    alpha=args.alpha,
                    heldout_calibration_target=args.calibration_target,
                ),
                common,
                settings,
                role_settings={
                    sensor: temporal_settings(dataset, sensor)
                    for sensor in cfg["b_cols"]
                },
            )
        )
    return rows


def train_toy_reference(seed, device):
    cfg = dict(
        base_config("toy_spatial"), seed=seed, anchor_sensor="sensor_0", anchor_idx=0
    )
    ensure_data("toy_spatial", cfg)
    frame, cfg = load_dataset("toy_spatial", cfg)
    fold = prepare_toy_fold(frame, cfg, device)
    return fit_fold(fold, device), fold


def toy_rows(args, seed):
    device = initialize_runtime(seed)
    model, fold = train_toy_reference(seed, device)
    cfg = fold.config
    cfg["heldout_raw_scale"] = args.raw_scale
    predictions = _predictions(
        model, fold.pool, cfg, correct_surrogate=args.calibration_target != "raw_sensor"
    )
    frame, loaded = load_dataset("toy_spatial", cfg)
    from ncam.experiments.preparation import toy_split_frames

    _, pool = toy_split_frames(frame, loaded)
    rows = []
    for split, (cal_idx, test_idx) in enumerate(
        build_random_split_indices(
            len(pool), int(cfg["n_splits"]), float(cfg["cv_test_size"]), seed
        )
    ):
        cal, test = (
            {
                role: {field: values[idx] for field, values in fields.items()}
                for role, fields in predictions.items()
            }
            for idx in (cal_idx, test_idx)
        )
        common = {
            "dataset": "toy_spatial",
            "seed": seed,
            "origin": np.nan,
            "split": split,
            "anchor_sensor": "sensor_0",
        }
        rows.extend(
            _partition_rows(
                cal,
                test,
                pool.iloc[cal_idx],
                pool.iloc[test_idx],
                dict(
                    cfg,
                    alpha=args.alpha,
                    heldout_calibration_target=args.calibration_target,
                ),
                common,
            )
        )
    return rows


def write_results(args, raw):
    raw_sensor = getattr(args, "calibration_target", None) == "raw_sensor"
    raw_scale = getattr(args, "raw_scale", "constant")
    adaptive = raw_sensor and raw_scale != "constant"
    sensor_method = (
        ADAPTIVE_RAW_SENSOR_CP if adaptive else RAW_SENSOR_CP if raw_sensor else HS_CP
    )
    keys = ["dataset", "method", "held_out_sensor", "seed"]
    if raw.duplicated([*keys, "origin", "split"]).any():
        raise ValueError("Duplicate experiment results")
    expected_roles = {(POSTERIOR_PREDICTIVE, ""), (ORACLE, "")}
    for dataset in args.datasets:
        expected = expected_roles | {
            (sensor_method, s) for s in DATASET_SENSORS[dataset]
        }
        group = raw.loc[raw.dataset == dataset]
        if set(group.seed) != set(args.seeds):
            raise ValueError("Incomplete experiment seeds")
        partition_key = "split" if dataset == "toy_spatial" else "origin"
        expected_partitions = set(group[partition_key])
        for _, runs in group.groupby("seed"):
            if set(runs[partition_key]) != expected_partitions:
                raise ValueError("Incomplete experiment partitions within seed")
        for _, partition in group.groupby(["seed", "origin", "split"], dropna=False):
            observed = set(
                zip(partition.method, partition.held_out_sensor, strict=True)
            )
            if observed != expected or partition.test_rows.nunique() != 1:
                raise ValueError("Incomplete roles or different evaluation rows")
    per_seed, summary = summarize_results(raw)
    extra = []
    for identity, group in raw.groupby(keys, sort=False):
        values = group.coverage_raw_sensor_row.to_numpy()
        extra.append(
            dict(
                zip(keys, identity, strict=True),
                coverage_raw_sensor_row=(
                    float(np.average(values, weights=group.test_rows))
                    if group.origin.notna().any()
                    else float(values.mean())
                ),
            )
        )
    per_seed = per_seed.merge(pd.DataFrame(extra), on=keys, validate="one_to_one")
    raw_summary = per_seed.groupby(keys[:-1]).coverage_raw_sensor_row.agg(
        ["mean", "std"]
    )
    raw_summary.columns = [f"coverage_raw_sensor_row_{s}" for s in raw_summary.columns]
    summary = summary.merge(
        raw_summary.reset_index(), on=keys[:-1], validate="one_to_one"
    )
    table = []
    for row in summary.to_dict("records"):
        item = {"Dataset": DATASET_LABELS[row["dataset"]], "Interval": row["method"]}
        item["Held-out sensor"] = row["held_out_sensor"] or "—"
        measures = (
            ("Reference coverage", "coverage_reference_row"),
            (
                (
                    "Held-out sensor coverage"
                    if raw_sensor
                    else "Corrected sensor coverage"
                ),
                "coverage_sensor_row",
            ),
            ("Raw sensor coverage", "coverage_raw_sensor_row"),
            ("Width", "width"),
        )
        for title, measure in measures:
            if raw_sensor and title == "Raw sensor coverage":
                continue
            mean, std = row[f"{measure}_mean"], row[f"{measure}_std"]
            item[title] = (
                "—"
                if not np.isfinite(mean)
                else f"{mean:.4f} ± {std:.4f}" if row["n_seeds"] > 1 else f"{mean:.4f}"
            )
        table.append(item)
    args.output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output / "results.csv", index=False)
    per_seed.to_csv(args.output / "per_seed.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(table).to_csv(args.output / "table.csv", index=False)
    pd.DataFrame(table).to_latex(args.output / "table.tex", index=False, escape=True)
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "training_sensors": "all three; frozen Table 2 reference checkpoints",
                "fusion_sensors": (
                    "exclude calibration sensor on both calibration and test"
                ),
                "surrogate": (
                    "uncorrected raw sensor reading"
                    if raw_sensor
                    else "(transformed_sensor - learned_offset) / learned_gain"
                ),
                "score": (
                    "abs(raw_sensor_reading - raw_posterior_median)"
                    + (" / raw_predictive_spread" if adaptive else "")
                    if raw_sensor
                    else (
                        "abs(surrogate - posterior_mean) / "
                        "sqrt(posterior_variance + sigma_y^2)"
                    )
                ),
                "temporal_settings": (
                    "inherit frozen block/window/scheme; "
                    + (
                        "normalized raw-space sensor score"
                        if adaptive
                        else (
                            "absolute raw-space sensor score"
                            if raw_sensor
                            else "normalized model-space score"
                        )
                    )
                ),
                "coverage_sensor_row": (
                    "uncorrected sensor reading in raw interval"
                    if raw_sensor
                    else "bias-corrected surrogate in model space"
                ),
                "coverage_raw_sensor_row": "unadjusted sensor reading in raw interval",
                "coverage_reference_row": "reference label in raw interval",
                "aggregation": (
                    "pool calendar rows within seed; "
                    "average Toy partitions; seed mean/std"
                ),
                "alpha": args.alpha,
                "seeds": args.seeds,
                "reserved_holdout_evaluated": False,
                "calibration_settings_retuned": bool(
                    getattr(args, "heldout_profile", None)
                ),
                "reference_labels_used_for_tuning": False,
                "heldout_calibration_target": (
                    "raw_sensor" if raw_sensor else "bias_corrected_sensor"
                ),
                "heldout_sensor_correction_applied": not raw_sensor,
                "raw_sensor_settings_retuned": False,
                "raw_scale": raw_scale if raw_sensor else "not_applicable",
                "raw_scale_definition": (
                    "half raw predictive quantile span at Phi(-1), Phi(1); "
                    "fixed floor 1e-8; sensor variance = gain^2 * posterior_variance "
                    "+ learned_sensor_variance; posterior scale excludes sigma_y"
                    if adaptive
                    else "constant scale of one" if raw_sensor else None
                ),
                "adaptive_scale_selected": raw_scale == "posterior" and raw_sensor,
                "adaptive_scale_selection": (
                    "user chose posterior after sensor-only development comparison"
                    if raw_scale == "posterior" and raw_sensor
                    else None
                ),
                "temporal_validity": (
                    "requires suitable dependence/invariance assumptions"
                ),
                "point_prediction_summary": "median",
                "heldout_profile": (
                    str(args.heldout_profile.resolve())
                    if getattr(args, "heldout_profile", None)
                    else None
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        pd.DataFrame(table).to_markdown(index=False, disable_numparse=True), flush=True
    )


def main():
    args = parse_args(
        __file__, __doc__, seeds=(0, 1, 42), configure_parser=configure_parser
    )
    if not np.isfinite(args.alpha) or not 0 < args.alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if args.raw_scale != "constant" and args.calibration_target != "raw_sensor":
        raise ValueError("Adaptive raw scales require --calibration-target raw_sensor")
    if args.calibration_target != "raw_sensor":
        args.output = args.output / "bias_corrected_sensor"
    elif args.raw_scale != "constant":
        args.output = args.output / f"adaptive_{args.raw_scale}"
    # Inference is inexpensive; one thread avoids CPU oversubscription on Toy.
    torch.set_num_threads(1)
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            rows.extend(
                toy_rows(args, seed)
                if dataset == "toy_spatial"
                else calendar_rows(args, dataset, seed)
            )
    write_results(args, pd.DataFrame(rows))


if __name__ == "__main__":
    main()
