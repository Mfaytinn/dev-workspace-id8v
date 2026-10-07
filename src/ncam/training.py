"""Fresh sensor-only training with validation-NLL model selection."""

import copy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from ncam import loss
from ncam.evaluation import posterior_point_prediction


def _run_train_batches(
    model,
    train_loader,
    optimizer,
    device,
    var_reg_weight: float,
    grad_clip_norm: float,
    epoch: int,
):
    model.train()
    total_loss_sum = nll_loss_sum = 0.0
    row_count = 0

    for batch_x, batch_b in train_loader:
        batch_x = batch_x.to(device)
        batch_b = batch_b.to(device)

        optimizer.zero_grad()
        mu_0, log_var_0, log_var_j, mult_bias, add_bias = model(batch_x)

        nll_loss = loss.nll_marginal_sensors(
            batch_b,
            mu_0,
            log_var_0,
            log_var_j,
            mult_bias,
            add_bias,
            variance_transform=getattr(model, "variance_transform", "softplus"),
        )

        _, var_reg_weighted = loss.compute_regularization(
            log_var_0, log_var_j, var_reg_weight
        )

        total_loss = nll_loss + var_reg_weighted

        if not torch.isfinite(total_loss):
            print(
                f"Non-finite training loss at epoch {epoch + 1}; "
                "failing this training run."
            )
            return {}, True

        total_loss.backward()

        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        grads_finite = True
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grads_finite = False
                break
        if not grads_finite:
            print(
                f"Non-finite gradients at epoch {epoch + 1}; failing this training run."
            )
            return {}, True

        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            return {}, True
        nll_loss_sum += float(nll_loss.item()) * len(batch_b)
        total_loss_sum += float(total_loss.item()) * len(batch_b)
        row_count += len(batch_b)

    if row_count == 0:
        return {}, True

    return {
        "avg_train_nll": nll_loss_sum / row_count,
        "avg_train_total_loss": total_loss_sum / row_count,
    }, False


def train_ncam_model(model, train_loader, X_val, B_val, config, device):
    """Train from scratch; reference labels never enter this function."""
    epochs = int(config["epochs"])
    if epochs < 1:
        raise ValueError("Training requires a positive epoch budget")
    if config.get("bias_identity_weight", 0) != 0:
        raise ValueError("NCAM uses only sensor NLL and variance regularization")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    for group in optimizer.param_groups:
        group["base_lr"] = float(group["lr"])
    warmup = int(config["warmup_epochs"])
    scheduler = build_scheduler(
        optimizer, float(config["scheduler_min_lr"]), epochs, warmup
    )
    best_nll, best_state, waiting = float("inf"), None, 0
    history = {
        "epochs": [],
        "train_nll": [],
        "val_loss": [],
        "validation_predictions": [],
    }
    for epoch in range(epochs):
        if warmup > 0 and epoch < warmup:
            fraction = epoch / max(warmup - 1, 1)
            for group in optimizer.param_groups:
                base = group["base_lr"]
                group["lr"] = base / 10 + (base - base / 10) * fraction
        metrics, failed = _run_train_batches(
            model,
            train_loader,
            optimizer,
            device,
            float(config["var_reg_weight"]),
            float(config["grad_clip_norm"]),
            epoch,
        )
        if failed:
            raise RuntimeError(
                f"Nonfinite training loss, gradients or parameters at epoch {epoch + 1}"
            )
        model.eval()
        with torch.no_grad():
            params = model(X_val)
            nll = loss.nll_marginal_sensors(
                B_val, *params, variance_transform=model.variance_transform
            )
            _, regularization = loss.compute_regularization(
                params[1], params[2], float(config["var_reg_weight"])
            )
            if not torch.isfinite(nll + regularization):
                raise RuntimeError(
                    f"Nonfinite validation sensor NLL at epoch {epoch + 1}"
                )
            if config.get("record_validation_predictions", False):
                mean, variance = model.predict_posterior(X_val, B_val)
                point = posterior_point_prediction(
                    mean.ravel(), variance.ravel(), config["obs_transform"]
                )
                history["validation_predictions"].append(point.cpu().numpy())
        val_nll = float(nll)
        history["epochs"].append(epoch + 1)
        history["train_nll"].append(metrics["avg_train_nll"])
        history["val_loss"].append(val_nll)
        if warmup == 0 or epoch >= warmup:
            scheduler.step()
        if val_nll < best_nll:
            best_nll, best_state, waiting = (
                val_nll,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            waiting += 1
            if int(config["patience"]) > 0 and waiting >= int(config["patience"]):
                break
        if (epoch + 1) % 50 == 0 or epoch + 1 == epochs:
            print(
                f"Epoch {epoch + 1}/{epochs}: train NLL {metrics['avg_train_nll']:.4f}, validation NLL {val_nll:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    return model, best_nll, history


def build_scheduler(optimizer, scheduler_min_lr, epochs, warmup_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=scheduler_min_lr
    )


class SensorDataset(Dataset):
    def __init__(self, x: torch.Tensor, b: torch.Tensor) -> None:
        self.x = x.float()
        self.b = b.float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.b[idx]


def create_data_loaders(
    x_train: torch.Tensor | np.ndarray,
    b_train: torch.Tensor | np.ndarray,
    x_val: torch.Tensor | np.ndarray,
    b_val: torch.Tensor | np.ndarray,
    batch_size: int,
    device: str,
    seed: int,
    shuffle_train: bool = True,
) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    b_train_t = torch.as_tensor(b_train, dtype=torch.float32)
    x_val_arr = torch.as_tensor(x_val, dtype=torch.float32)
    b_val_arr = torch.as_tensor(b_val, dtype=torch.float32)

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_loader = DataLoader(
        SensorDataset(x_train_t, b_train_t),
        batch_size=batch_size,
        shuffle=bool(shuffle_train),
        pin_memory=device == "cuda",
        generator=generator,
    )
    x_val_t = x_val_arr.to(device=device, dtype=torch.float32)
    b_val_t = b_val_arr.to(device=device, dtype=torch.float32)
    return train_loader, x_val_t, b_val_t


def initialize_runtime(seed: int) -> str:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    torch.manual_seed(seed)
    print(f"Using device: {device}")
    return device
