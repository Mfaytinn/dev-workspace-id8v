from typing import NamedTuple

import numpy as np
import torch

from ncam.calibration import (
    predict_interval,
    split_conformal,
)


class Metrics(NamedTuple):
    """Raw-target-scale metrics, serialized with the existing CSV names."""

    coverage: float
    width: float
    rmse: float
    mae: float


def invert_obs_transform(y_values: torch.Tensor, obs_transform: str) -> torch.Tensor:
    """Invert model-space values or interval bounds to raw observation units."""
    transforms = {
        "none": torch.Tensor.float,
        "log": torch.Tensor.exp,
        "log1p": torch.Tensor.expm1,
    }
    return transforms[obs_transform](y_values)


def posterior_point_prediction(
    mean: torch.Tensor,
    variance: torch.Tensor,
    obs_transform: str,
    summary: str = "median",
) -> torch.Tensor:
    """Return the raw-space posterior median, independent of predictive variance."""
    if summary != "median":
        raise ValueError("Only posterior-median point predictions are supported")
    if obs_transform == "none":
        return mean
    if obs_transform not in {"none", "log", "log1p"}:
        raise ValueError("obs_transform must be 'none', 'log' or 'log1p'")
    return invert_obs_transform(mean, obs_transform)


def ground_truth_coverage(lower, upper, Y_test):
    """
    Calculates coverage based on whether the ground truth falls in the interval.

    Args:
        lower: torch.Tensor of lower bounds
        upper: torch.Tensor of upper bounds
        Y_test: torch.Tensor of ground truth values
    """
    in_interval = (Y_test >= lower) & (Y_test <= upper)
    return float(torch.mean(in_interval.float()).item())


def timestamp_joint_coverage(lower, upper, targets, timestamps) -> float:
    """Fraction of timestamps with all observed locations covered.

    This is joint across locations within one timestamp, not across the
    full test window of timestamps.
    """
    hit = ((targets >= lower) & (targets <= upper)).detach().cpu().numpy()
    times, inverse = np.unique(np.asarray(timestamps), return_inverse=True)
    covered = np.ones(len(times), dtype=bool)
    np.logical_and.at(covered, inverse, hit)
    return float(covered.mean())


def _log_floor_for_targets(config: dict) -> float:
    """Select a deterministic log floor for transforming Y into model space.

    For obs_transform='log', calibration must compare mu_post (model space) to Y
    in the *same* space. We keep the floor independent of calibration labels to
    preserve exchangeability: prefer an explicit config value, otherwise derive
    a global floor from the per-sensor training floors.
    """

    if "log_floor_y" in config and config["log_floor_y"] is not None:
        floor = float(config["log_floor_y"])
    else:
        per_sensor = config.get("log_floor_per_sensor")
        arr = np.asarray(per_sensor, dtype=np.float64).reshape(-1)
        floor = float(np.min(arr))
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("Target log floor must be finite and positive")
    return floor


def _transform_targets_to_model_space(
    y_raw_t: torch.Tensor, config: dict
) -> torch.Tensor:
    """Transform raw-scale targets Y into the model's observation space."""

    obs_transform = str(config.get("obs_transform", "none"))
    y_raw_t = y_raw_t.to(dtype=torch.float32).ravel()
    transforms = {
        "none": lambda: y_raw_t,
        "log": lambda: torch.log(
            torch.clamp(y_raw_t, min=float(_log_floor_for_targets(config)))
        ),
        "log1p": lambda: torch.log1p(y_raw_t),
    }
    return transforms[obs_transform]()


def predict_intervals(X_test_t, B_test_t, model, tau, config):
    """Return the raw posterior point prediction and transformed interval bounds."""
    _, lower, upper = predict_interval(
        X_test_t,
        B_test_t,
        model,
        tau,
        sigma_y=config["sigma_y"],
    )
    device = X_test_t.device
    obs_transform = str(config.get("obs_transform", "none"))
    raw_lower, raw_upper = (
        invert_obs_transform(
            torch.as_tensor(values, device=device, dtype=torch.float32).ravel(),
            obs_transform,
        )
        for values in (lower, upper)
    )
    with torch.no_grad():
        mean, variance = model.predict_posterior(X_test_t, B_test_t)
        center = posterior_point_prediction(
            mean.ravel(),
            variance.ravel(),
            obs_transform,
            config.get("point_prediction_summary", "median"),
        )
    if obs_transform == "log":
        floor = _log_floor_for_targets(config)
        log_floor = float(np.log(np.float32(floor)))
        model_lower = torch.as_tensor(lower, device=device)
        model_upper = torch.as_tensor(upper, device=device)
        raw_lower = torch.where(model_lower <= log_floor, 0.0, raw_lower)
        empty = model_upper < log_floor
        raw_lower = torch.where(empty, floor, raw_lower)
        raw_upper = torch.where(empty, 0.0, raw_upper)
        return center, raw_lower, raw_upper
    return center, raw_lower, raw_upper


def run_evaluation_suite(
    X_cal_t,
    B_cal_t,
    Y_cal_t,
    X_test_t,
    B_test_t,
    Y_test_raw_t,
    model,
    config,
):
    """
    Run ground-truth split calibration and return its raw-scale metrics.

    All inputs are PyTorch tensors. B_cal_t/B_test_t are in model space (raw or
    transformed); metrics and coverage are always computed in raw scale.
    """
    # References and conformity scores use model space; reported metrics use raw units.
    Y_cal_model_t = _transform_targets_to_model_space(Y_cal_t, config)
    tau_split = split_conformal(
        X_cal_t,
        B_cal_t,
        Y_cal_model_t,
        model,
        alpha=config["alpha"],
        sigma_y=config["sigma_y"],
        normalize_scores=True,
    )
    mu, lower, upper = predict_intervals(X_test_t, B_test_t, model, tau_split, config)
    metrics = Metrics(
        ground_truth_coverage(lower, upper, Y_test_raw_t),
        float(torch.mean((upper - lower).clamp_min(0)).item()),
        float(torch.sqrt(torch.mean((Y_test_raw_t - mu) ** 2)).item()),
        float(torch.mean(torch.abs(Y_test_raw_t - mu)).item()),
    )
    return {"Split CP (gt)": metrics}
