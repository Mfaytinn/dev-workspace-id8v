"""Point prediction comparisons on a fixed held-out split."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ncam.baselines import (
    GaussianFusionBaseline,
    InverseVarianceWeightedMean,
    KalmanFilterBaseline,
    PPCABaseline,
    SensorMeanBaseline,
)
from ncam.evaluation import invert_obs_transform, posterior_point_prediction
from ncam.experiments.records import PerformanceRow
from ncam.veli import VeliBaseline

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_BASELINE_SPECS = [
    ("Gaussian fusion baseline", "gaussian"),
    ("Inverse variance weighting baseline", "ivw"),
    ("Mean module baseline", "mean"),
    ("Probabilistic PCA baseline", "ppca"),
]

VELI_BASELINE_SPEC = ("Veli baseline", "veli")


def run_baseline_comparison(
    model: torch.nn.Module,
    x_test_t: torch.Tensor,
    b_test_t: torch.Tensor,
    y_test: torch.Tensor | np.ndarray,
    b_test_raw: torch.Tensor | np.ndarray,
    b_train_model: torch.Tensor | np.ndarray,
    sensor_names: list[str],
    device: str,
    cfg: dict,
    obs_transform: str,
    test_keys: pd.DataFrame | None = None,
    all_prediction_csv_path: str | None = None,
    train_keys: pd.DataFrame | None = None,
) -> list[tuple[str, float, float]]:
    ncam = cast("Any", model)
    point_summary = str(cfg.get("point_prediction_summary", "median"))

    def to_obs_space(mean_t: torch.Tensor) -> torch.Tensor:
        mean_use = mean_t.squeeze(-1)
        return invert_obs_transform(mean_use, obs_transform)

    with torch.no_grad():
        mu_test, var_test = ncam.predict_posterior(x_test_t, b_test_t)
        y_pred = posterior_point_prediction(
            mu_test.squeeze(-1), var_test.squeeze(-1), obs_transform, point_summary
        )

    y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred
    model_rmse = float(np.sqrt(np.mean((y_test_np - y_pred_np) ** 2)))
    model_mae = float(np.mean(np.abs(y_test_np - y_pred_np)))
    all_predictions = None
    if all_prediction_csv_path is not None:
        if test_keys is None or len(test_keys) != len(y_test_np):
            raise ValueError("Prediction export requires matching test row keys")
        all_predictions = test_keys.reset_index(drop=True).copy()
        all_predictions["reference"] = y_test_np
        all_predictions[cfg.get("model", "Model")] = y_pred_np
        all_predictions["anchored_median"] = y_pred_np

    rows: list[tuple[str, float, float]] = [
        (cfg.get("model", "Model"), model_rmse, model_mae)
    ]

    specs = list(DEFAULT_BASELINE_SPECS) if cfg.get("baselines", True) else []
    temporal = (
        cfg.get("dataset") in {"seneurcity", "cairsense"}
        or cfg.get("evaluation_protocol") == "calendar"
    )
    if temporal and specs:
        specs[0] = ("Temporal Kalman filter baseline", "kalman")
    if cfg.get("baselines", True) and cfg.get("veli"):
        specs.append(VELI_BASELINE_SPEC)
    b_train_fit = (
        b_train_model.float().to(device)
        if isinstance(b_train_model, torch.Tensor)
        else torch.as_tensor(b_train_model, device=device, dtype=torch.float32)
    )
    n_sensors = len(sensor_names)

    veli_kwargs = dict(cfg.get("veli") or {})
    veli_kwargs.setdefault("obs_transform", obs_transform)
    if "seed" not in veli_kwargs and "seed" in cfg:
        veli_kwargs["seed"] = int(cfg["seed"])
    if "anchor_idx" not in veli_kwargs and "anchor_idx" in cfg:
        veli_kwargs["anchor_idx"] = int(cfg["anchor_idx"])
    baseline_builders = {
        "kalman": lambda: KalmanFilterBaseline(n_sensors=n_sensors),
        "gaussian": lambda: GaussianFusionBaseline(n_sensors=n_sensors),
        "ivw": lambda: InverseVarianceWeightedMean(n_sensors=n_sensors),
        "mean": lambda: SensorMeanBaseline(n_sensors=n_sensors),
        "ppca": lambda: PPCABaseline(n_sensors=n_sensors),
        "veli": lambda: VeliBaseline(n_sensors=n_sensors, **veli_kwargs),
    }

    for display_name, baseline_type in specs:
        baseline = baseline_builders[baseline_type]().to(device)
        baseline_ncam = cast("Any", baseline)
        if isinstance(baseline, KalmanFilterBaseline):
            baseline.fit(b_train_fit, train_keys)
        elif hasattr(baseline_ncam, "fit"):
            baseline_ncam.fit(b_train_fit)
        baseline.eval()

        with torch.no_grad():
            if isinstance(baseline, KalmanFilterBaseline):
                mu_baseline, var_baseline = baseline.predict_posterior(
                    x_test_t, b_test_t, test_keys
                )
            else:
                mu_baseline, var_baseline = baseline_ncam.predict_posterior(
                    x_test_t, b_test_t
                )
            if getattr(baseline, "posterior_in_obs_space", False):
                y_baseline_pred = mu_baseline.squeeze(-1)
            elif isinstance(
                baseline, (KalmanFilterBaseline, GaussianFusionBaseline, PPCABaseline)
            ):
                y_baseline_pred = posterior_point_prediction(
                    mu_baseline.squeeze(-1),
                    var_baseline.squeeze(-1),
                    obs_transform,
                    point_summary,
                )
            else:
                # Mean and inverse-variance fusion expose variance proxies, not
                # Gaussian predictive posteriors; retain their point estimates.
                y_baseline_pred = to_obs_space(mu_baseline)
        y_baseline_pred_np = (
            y_baseline_pred.cpu().numpy()
            if isinstance(y_baseline_pred, torch.Tensor)
            else y_baseline_pred
        )
        baseline_rmse = float(np.sqrt(np.mean((y_test_np - y_baseline_pred_np) ** 2)))
        baseline_mae = float(np.mean(np.abs(y_test_np - y_baseline_pred_np)))
        rows.append((display_name, baseline_rmse, baseline_mae))
        if all_predictions is not None:
            all_predictions[display_name] = y_baseline_pred_np

    b_test_raw_np = (
        b_test_raw.cpu().numpy() if isinstance(b_test_raw, torch.Tensor) else b_test_raw
    )
    for idx, sensor_name in enumerate(
        sensor_names if cfg.get("baselines", True) else []
    ):
        sensor_pred = b_test_raw_np[:, idx]
        sensor_rmse = float(np.sqrt(np.mean((y_test_np - sensor_pred) ** 2)))
        sensor_mae = float(np.mean(np.abs(y_test_np - sensor_pred)))
        rows.append(
            (f"Single-sensor baseline ({sensor_name})", sensor_rmse, sensor_mae)
        )
        if all_predictions is not None:
            all_predictions[f"Single-sensor baseline ({sensor_name})"] = sensor_pred

    if all_predictions is not None:
        output = Path(all_prediction_csv_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        all_predictions.to_csv(output, index=False)

    return [PerformanceRow(*row) for row in rows]
