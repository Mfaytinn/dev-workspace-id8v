"""Fit paper models and choose candidates using sensor validation NLL."""

from dataclasses import replace

import torch

from ncam.checkpoints import save_model
from ncam.experiments.config import (
    configured_anchor,
    create_model,
    validate_model_candidates,
    validate_prior_initialization,
)
from ncam.training import create_data_loaders, train_ncam_model


def initialize_sensor_prior(model, train_x, train_sensors, config: dict) -> dict:
    """Initialize only the contextual mean head using transformed train sensors.

    All layers remain trainable. References and non-training partitions are absent here.
    """
    validate_prior_initialization(config)
    mode = config.get("prior_initialization", "random")
    if mode == "random":
        return {"mode": mode}
    anchor_index = (
        model.anchor_idx
        if hasattr(model, "anchor_idx")
        else configured_anchor(
            config["b_cols"], config.get("anchor_sensor", config["b_cols"][0])
        )[1]
    )
    target = train_sensors[:, anchor_index].detach().cpu()
    if not len(target) or not torch.isfinite(target).all():
        raise ValueError("Prior initialization requires finite training anchor values")
    device = next(model.parameters()).device
    head = model.prior_net.net[-1]
    with torch.no_grad():
        head.weight[0].zero_()
        head.bias[0] = target.mean().to(device)
    return {"mode": mode, "target_mean": float(target.mean())}


def _fit(fold, device):
    cfg = fold.config
    torch.manual_seed(int(cfg["seed"]))
    model = create_model(cfg, len(cfg["x_cols"]), cfg["b_cols"]).to(device)
    initialize_sensor_prior(model, fold.train.x, fold.train.sensors, cfg)
    loader, xv, bv = create_data_loaders(
        fold.train.x,
        fold.train.sensors,
        fold.val.x,
        fold.val.sensors,
        int(cfg["batch_size"]),
        device,
        int(cfg["seed"]),
        shuffle_train=bool(cfg["shuffle_train"]),
    )
    return train_ncam_model(model, loader, xv, bv, cfg, device)


def fit_fold(fold, device, *, save_path=None):
    """Train fresh and keep the resolved configuration of the NLL winner."""
    candidates = fold.config.get("model_candidates") or {}
    validate_model_candidates(candidates)
    if candidates:
        choices = []
        for name, overrides in sorted(candidates.items()):
            candidate = replace(
                fold, config=dict(fold.config, **overrides, model_candidates={})
            )
            model, nll, history = _fit(candidate, device)
            choices.append((nll, name, model, history, candidate.config))
        _, _, model, history, cfg = min(choices, key=lambda item: (item[0], item[1]))
        fold.config = cfg
    else:
        model, _, history = _fit(fold, device)
    model.eval()
    if save_path is not None:
        save_model(save_path, model, fold.config)
    model.training_history = history
    return model
