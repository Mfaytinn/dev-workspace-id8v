"""Temporal CP settings shared by tuning and deployment evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ncam.calibration import temporal_block_conformal, temporal_calibration_metadata
from ncam.evaluation import posterior_point_prediction


@dataclass(frozen=True)
class TemporalCPSettings:
    block_size: int = 12
    window_timestamps: int = 0  # Zero keeps the entire calibration partition.
    scale_power: float = 1.0
    scheme: str = "nob"  # Non-overlapping block permutations.

    def __post_init__(self):
        for name, minimum in (("block_size", 1), ("window_timestamps", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.scale_power) or not 0 <= self.scale_power <= 1:
            raise ValueError("scale_power must lie in [0, 1]")
        temporal_calibration_metadata(
            0, alpha=0.1, block_size=self.block_size, scheme=self.scheme
        )

    def to_dict(self) -> dict:
        return asdict(self)


def calibration_window(timestamps, settings: TemporalCPSettings) -> np.ndarray:
    """Keep whole recent observed timestamps, including every location."""
    timestamps = np.asarray(timestamps)
    times = np.unique(timestamps)
    if times.size == 0:
        raise ValueError("Calibration timestamps are empty")
    count = settings.window_timestamps
    start = times[-count] if count and count < times.size else times[0]
    return timestamps >= start


def score_scale(std, settings: TemporalCPSettings) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32).ravel()
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Predictive standard deviations must be finite and positive")
    return std**settings.scale_power


def temporal_threshold(mean, std, target, timestamps, settings, *, alpha=0.1):
    """Calibrate in model space; the score scale is also used in prediction."""
    mask = calibration_window(timestamps, settings)
    scores = np.abs(
        np.asarray(target).ravel() - np.asarray(mean).ravel()
    ) / score_scale(std, settings)
    return temporal_block_conformal(
        scores[mask],
        np.asarray(timestamps)[mask],
        alpha=alpha,
        block_size=settings.block_size,
        scheme=settings.scheme,
    )


def temporal_bounds(
    mean,
    std,
    threshold,
    settings,
    *,
    obs_transform,
    target_floor=None,
    lower_support=None,
):
    """Return raw-unit bounds with the same scale used for calibration scores."""
    mean = np.asarray(mean, dtype=np.float32).ravel()
    radius = float(threshold) * score_scale(std, settings)
    lower, upper = mean - radius, mean + radius
    if obs_transform == "log":
        model_lower, model_upper = lower.copy(), upper.copy()
        with np.errstate(over="ignore"):
            lower, upper = np.exp(lower), np.exp(upper)
        if target_floor is not None:
            if not np.isfinite(target_floor) or target_floor <= 0:
                raise ValueError("target_floor must be finite and positive")
            # Inverse image of log(max(y, floor)), on nonnegative PM support.
            log_floor = np.log(np.float32(target_floor))
            lower = np.where(model_lower <= log_floor, 0.0, lower)
            empty = model_upper < log_floor
            lower = np.where(empty, target_floor, lower)
            upper = np.where(empty, 0.0, upper)
    elif obs_transform == "log1p":
        with np.errstate(over="ignore"):
            lower, upper = np.expm1(lower), np.expm1(upper)
    elif obs_transform != "none":
        raise ValueError(f"Unsupported observation transform: {obs_transform}")
    if lower_support is not None:
        if not np.isfinite(lower_support):
            raise ValueError("lower_support must be finite")
        lower = np.maximum(lower, lower_support)
    return lower, upper


def timestamp_coverage(hit, timestamps):
    times, inverse = np.unique(timestamps, return_inverse=True)
    covered = np.ones(len(times), dtype=bool)
    np.logical_and.at(covered, inverse, hit)
    return float(covered.mean())


def evaluate_temporal_predictions(
    predictions,
    settings,
    *,
    alpha,
    obs_transform,
    target_floor=None,
    lower_support=None,
):
    """Evaluate cached predictions without accessing model inputs or training."""
    threshold = temporal_threshold(
        predictions["cal_mean"],
        predictions["cal_std"],
        predictions["cal_target"],
        predictions["cal_times"],
        settings,
        alpha=alpha,
    )
    used = calibration_window(predictions["cal_times"], settings)
    count = len(np.unique(predictions["cal_times"][used]))
    common = {
        **settings.to_dict(),
        "threshold": threshold,
        "cal_rows": len(used),
        "cal_rows_used": int(used.sum()),
        "cal_timestamps": len(np.unique(predictions["cal_times"])),
        "cal_timestamps_used": count,
        **temporal_calibration_metadata(
            count, alpha=alpha, block_size=settings.block_size, scheme=settings.scheme
        ),
        "test_rows": len(predictions["test_times"]),
        "test_timestamps": len(np.unique(predictions["test_times"])),
    }
    if not np.isfinite(threshold):
        return dict(common, valid=False)
    lower, upper = temporal_bounds(
        predictions["test_mean"],
        predictions["test_std"],
        threshold,
        settings,
        obs_transform=obs_transform,
        target_floor=target_floor,
        lower_support=lower_support,
    )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return dict(common, valid=False)
    return _interval_metrics(predictions, lower, upper, common, lower_support)


def _interval_metrics(predictions, lower, upper, common, lower_support):
    reference = predictions["test_reference"]
    target = predictions["test_target"]
    ref_hit = (reference >= lower) & (reference <= upper)
    hit = (target >= lower) & (target <= upper)
    has_sensor = bool(predictions["has_sensor"])
    return dict(
        common,
        valid=True,
        coverage_primary=timestamp_coverage(hit, predictions["test_times"]),
        coverage_reference_row=float(ref_hit.mean()),
        coverage_sensor_row=float(hit.mean()) if has_sensor else np.nan,
        coverage_reference_timestamp=timestamp_coverage(
            ref_hit, predictions["test_times"]
        ),
        coverage_sensor_timestamp=(
            timestamp_coverage(hit, predictions["test_times"]) if has_sensor else np.nan
        ),
        width=float(np.maximum(upper - lower, 0.0).mean(dtype=np.float64)),
        lower_support_fraction=(
            float((lower <= lower_support).mean())
            if lower_support is not None
            else float((lower == 0).mean())
        ),
    )


def evaluate_temporal_locations(
    predictions,
    settings,
    *,
    alpha,
    obs_transform,
    target_floor=None,
    lower_support=None,
):
    """Calibrate each location's time series for individual-row coverage.

    Each location uses the same permutation scheme, nominal level, and recent
    window. Thresholds use calibration targets only. All test rows are retained;
    missing calibration locations and unbounded thresholds are explicit failures.
    This targets one future observation per location, not simultaneous locations.
    """
    cal_locations = np.asarray(predictions["cal_locations"])
    test_locations = np.asarray(predictions["test_locations"])
    if len(cal_locations) != len(predictions["cal_times"]) or len(
        test_locations
    ) != len(predictions["test_times"]):
        raise ValueError("Location identifiers must match prediction rows")
    lower = np.empty(len(test_locations), dtype=np.float32)
    upper = np.empty_like(lower)
    used = np.zeros(len(cal_locations), dtype=bool)
    thresholds, metadata = [], []
    for location in np.unique(test_locations):
        cal = cal_locations == location
        test = test_locations == location
        if not cal.any():
            raise ValueError(f"No calibration observations for location {location}")
        times = np.asarray(predictions["cal_times"])[cal]
        local_used = calibration_window(times, settings)
        used[np.flatnonzero(cal)[local_used]] = True
        threshold = temporal_threshold(
            np.asarray(predictions["cal_mean"])[cal],
            np.asarray(predictions["cal_std"])[cal],
            np.asarray(predictions["cal_target"])[cal],
            times,
            settings,
            alpha=alpha,
        )
        if not np.isfinite(threshold):
            return {"valid": False, "invalid_location": str(location)}
        lower[test], upper[test] = temporal_bounds(
            np.asarray(predictions["test_mean"])[test],
            np.asarray(predictions["test_std"])[test],
            threshold,
            settings,
            obs_transform=obs_transform,
            target_floor=target_floor,
            lower_support=lower_support,
        )
        thresholds.append(threshold)
        metadata.append(
            temporal_calibration_metadata(
                len(np.unique(times[local_used])),
                alpha=alpha,
                block_size=settings.block_size,
                scheme=settings.scheme,
            )
        )
    if not thresholds or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        return {"valid": False}
    common = {
        **settings.to_dict(),
        "coverage_unit": "per_location",
        "n_locations": len(thresholds),
        # A summary of location thresholds, not one shared prediction radius.
        "threshold": float(np.mean(thresholds)),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
        "permutations_min": min(item["permutations"] for item in metadata),
        "permutations_max": max(item["permutations"] for item in metadata),
        "cal_rows": len(cal_locations),
        "cal_rows_used": int(used.sum()),
        "cal_timestamps": len(np.unique(predictions["cal_times"])),
        "cal_timestamps_used": len(
            np.unique(np.asarray(predictions["cal_times"])[used])
        ),
        "test_rows": len(test_locations),
        "test_timestamps": len(np.unique(predictions["test_times"])),
    }
    result = _interval_metrics(predictions, lower, upper, common, lower_support)
    result["coverage_primary"] = result[
        (
            "coverage_sensor_row"
            if bool(predictions["has_sensor"])
            else "coverage_reference_row"
        )
    ]
    return result


def raw_posterior_point_predictions(predictions, obs_transform, summary="median"):
    """Convert cached model-space posteriors to explicitly chosen raw readouts."""
    raw = dict(predictions)
    for split in ("cal", "test"):
        mean = torch.as_tensor(predictions[split + "_mean"])
        key = split + "_variance"
        variance = torch.as_tensor(predictions.get(key, np.zeros_like(mean)))
        raw[split + "_mean"] = posterior_point_prediction(
            mean, variance, obs_transform, summary
        ).numpy()
        raw[split + "_std"] = np.ones_like(raw[split + "_mean"])
    return raw


def evaluate_raw_absolute(
    predictions,
    cal_target,
    *,
    alpha=0.1,
    scheme="nob",
    block_size=12,
    obs_transform="log",
    point_prediction_summary="median",
):
    """Evaluate temporal conformal intervals from absolute raw-unit residuals."""
    raw = raw_posterior_point_predictions(
        predictions, obs_transform, point_prediction_summary
    )
    raw["cal_target"] = np.asarray(cal_target, dtype=np.float32)
    return evaluate_temporal_predictions(
        raw,
        TemporalCPSettings(block_size=block_size, scale_power=0.0, scheme=scheme),
        alpha=alpha,
        obs_transform="none",
        lower_support=0.0,
    )
