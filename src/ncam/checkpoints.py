"""Save reference weights with the preprocessing needed to evaluate them."""

from pathlib import Path

import torch

from ncam.experiments.config import create_model


def save_model(path, model, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        key: value
        for key, value in config.items()
        if key
        not in {
            "record_validation_predictions",
            "coordinate_clipping",
        }
    }
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"state_dict": model.state_dict(), "config": cfg}, temporary)
    temporary.replace(path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = create_model(cfg, len(cfg["x_cols"]), cfg["b_cols"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg
