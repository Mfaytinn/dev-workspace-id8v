"""Read and validate the paper experiment configurations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ncam.models import (
    DEFAULT_OFFSET_PARAMETERIZATION,
    NCAMAnchored,
    NCAMNoBias,
    NCAMUnanchored,
)

if TYPE_CHECKING:
    import torch


def load_config(path: str | Path) -> dict:
    """Load a YAML mapping. Experiment variants belong in their own YAMLs."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    value.setdefault("variance_transform", "softplus")
    value.setdefault("point_prediction_summary", "median")
    if value.get("model") in {"anchored", "unanchored"}:
        value.setdefault("offset_parameterization", DEFAULT_OFFSET_PARAMETERIZATION)
    return value


MODEL_TUNING_KEYS = {
    "lr",
    "var_reg_weight",
    "weight_decay",
    "hidden_dim",
    "prior_hidden_dim",
    "rel_hidden_dim",
    "bias_hidden_dim",
    "batch_size",
}


def require_current_anchored(config: dict) -> None:
    """Require the current mean-residual calibration in both learned-bias models."""
    if (
        config.get("model") in {"anchored", "unanchored"}
        and config.get("offset_parameterization", DEFAULT_OFFSET_PARAMETERIZATION)
        != DEFAULT_OFFSET_PARAMETERIZATION
    ):
        raise ValueError("NCAM requires offset_parameterization: mean_residual")


def validate_model_candidates(candidates) -> None:
    """Keep candidate tuning confined to model and optimizer hyperparameters."""
    if candidates is None:
        return
    if not isinstance(candidates, dict):
        raise ValueError("model_candidates must be a mapping")
    for name, parameters in candidates.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("Model candidate names must be single directory names")
        if not isinstance(parameters, dict) or set(parameters) - MODEL_TUNING_KEYS:
            raise ValueError("Model candidates may change only tuning hyperparameters")
        for key, value in parameters.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Candidate {key} must be finite and numeric")
            if key.endswith("_dim") or key in {"batch_size"}:
                if not isinstance(value, int) or value < 1:
                    raise ValueError(f"Candidate {key} must be a positive integer")
            elif value < 0 or (key == "lr" and value == 0):
                raise ValueError(f"Invalid candidate {key}")


def configured_anchor(
    sensors: tuple[str, ...] | list[str], configured: str
) -> tuple[str, int]:
    """Resolve the configured anchor without changing the released sensor roles."""
    if not sensors:
        raise ValueError("Anchored NCAM requires at least one sensor")
    if configured not in sensors:
        raise ValueError("The configured anchor must be present in the input sensors")
    return configured, sensors.index(configured)


def validate_prior_initialization(config: dict) -> None:
    if config.get("prior_initialization", "random") not in {
        "random",
        "anchor_centered",
    }:
        raise ValueError("prior_initialization must be random or anchor_centered")
    if config.get("prior_initialization", "random") != "random" and config.get(
        "model"
    ) not in {"anchored", "nobias", "unanchored"}:
        raise ValueError("Sensor prior initialization requires an NCAM model")


def validate_experiment_config(config: dict) -> None:
    """Fail before training for unsupported choices or unresolved sweep values."""
    for key in (
        "model",
        "seed",
        "epochs",
        "batch_size",
        "shuffle_train",
    ):
        if key not in config:
            raise ValueError(f"Missing experiment configuration: {key}")
    if config["model"] not in {"nobias", "anchored", "unanchored"}:
        raise ValueError(f"Unsupported model: {config['model']}")
    require_current_anchored(config)
    if config.get("obs_transform") not in {"none", "log", "log1p"}:
        raise ValueError("obs_transform must be none, log or log1p")
    protocol = config.get("evaluation_protocol")
    if protocol not in {None, "calendar"}:
        raise ValueError("Real datasets support evaluation_protocol: calendar only")
    if "block_size_days" in config and protocol != "calendar":
        raise ValueError("Temporal datasets require evaluation_protocol: calendar")
    if protocol == "calendar":
        if not config.get("calendar_config"):
            raise ValueError("Calendar runs require a shared calendar_config")
        if not config.get("sensor_only_train"):
            raise ValueError("Calendar runs require sensor-only training")
        if config.get("refit_on_train_val", False):
            raise ValueError(
                "Calendar runs select a held-out validation NLL checkpoint"
            )
        if config.get("patience") != 0 or config.get("restore_best_model") is not True:
            raise ValueError(
                "Calendar runs require patience: 0 and restore_best_model: true"
            )
        fractions = [
            config.get(key, 0) for key in ("holdout_cal_frac", "holdout_test_frac")
        ]
        if any(not 0 < fraction < 1 for fraction in fractions) or sum(fractions) >= 1:
            raise ValueError(
                "Holdout fractions must be positive and sum to less than one"
            )
    if config.get("selection_metric", "val_nll") != "val_nll":
        raise ValueError("Training selection uses sensor validation NLL")
    validate_model_candidates(config.get("model_candidates"))
    for parameters in (config.get("model_candidates") or {}).values():
        require_current_anchored(dict(config, **parameters))
    for key, fixed in {
        "mlp_depth": 3,
        "mlp_activation": "gelu",
        "gain_parameterization": "direct",
        "feature_architecture": "dense",
        "prior_mean_parameterization": "location",
    }.items():
        if config.get(key, fixed) != fixed:
            raise ValueError(f"The paper models require {key}: {fixed}")
    if config.get("bias_identity_weight", 0.0) != 0:
        raise ValueError(
            "The paper objective permits only sensor NLL and variance regularization"
        )
    validate_prior_initialization(config)
    for key in ("epochs",):
        value = config.get(key, 10)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("var_reg_weight", "weight_decay"):
        value = config.get(key, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config.get("conformal_scheme", "nob") not in {"nob", "cyclic"}:
        raise ValueError("Unsupported temporal permutation scheme")
    if config.get("conformal_score_space", "model_scaled") not in {
        "raw_absolute",
        "model_absolute",
        "model_scaled",
        "model_normalized",
    }:
        raise ValueError("Unsupported conformal score space")
    if (
        config.get("conformal_scheme") == "cyclic"
        and config.get("conformal_block_size") != 1
    ):
        raise ValueError("All cyclic shifts require conformal_block_size: 1")
    if config.get("variance_transform", "softplus") != "softplus":
        raise ValueError("variance_transform must be softplus")
    if config.get("prior_mode", "contextual") != "contextual":
        raise ValueError("NCAM requires a contextual prior")
    if config.get("gain_mode", "learned") != "learned":
        raise ValueError("Anchored NCAM requires learned non-anchor gains")
    if config.get("point_prediction_summary", "median") != "median":
        raise ValueError("point_prediction_summary must be median")
    if not isinstance(config.get("clip_coordinates", False), bool):
        raise ValueError("clip_coordinates must be a boolean")
    if not isinstance(config.get("record_validation_predictions", False), bool):
        raise ValueError("record_validation_predictions must be a boolean")
    for key, value in config.items():
        if isinstance(value, list) and key not in {"x_cols", "b_cols"}:
            raise ValueError(f"Unresolved sweep axis: {key}")


def create_model(cfg: dict, in_dim: int, b_cols: list[str]) -> torch.nn.Module:
    """Build the fixed paper architecture with the resolved head widths."""
    if cfg.get("prior_mode", "contextual") != "contextual":
        raise ValueError("NCAM requires a contextual prior")
    if cfg.get("gain_mode", "learned") != "learned":
        raise ValueError("NCAM requires learned non-anchor gains")
    require_current_anchored(cfg)
    for key, fixed in {
        "mlp_depth": 3,
        "mlp_activation": "gelu",
        "gain_parameterization": "direct",
        "feature_architecture": "dense",
        "prior_mean_parameterization": "location",
    }.items():
        if cfg.get(key, fixed) != fixed:
            raise ValueError(f"The paper models require {key}: {fixed}")
    if cfg.get("bias_identity_weight", 0) != 0:
        raise ValueError("NCAM requires the paper-only objective")
    model_type = cfg["model"]
    model_class = {
        "anchored": NCAMAnchored,
        "nobias": NCAMNoBias,
        "unanchored": NCAMUnanchored,
    }[model_type]
    args = {
        "in_dim": in_dim,
        "n_sensors": len(b_cols),
        **{
            key: int(cfg[key])
            for key in (
                "hidden_dim",
                "prior_hidden_dim",
                "rel_hidden_dim",
                "bias_hidden_dim",
            )
        },
        "variance_transform": cfg.get("variance_transform", "softplus"),
    }
    if model_type == "anchored":
        args["anchor_idx"] = int(cfg["anchor_idx"])
    return model_class(**args)
