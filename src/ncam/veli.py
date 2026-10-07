"""NCAM adapter around the pinned authors' implementation in third_party/veli."""

from __future__ import annotations

import importlib
import random
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ncam.baselines import EPS, _BaselineBase

if TYPE_CHECKING:
    from types import ModuleType

UPSTREAM_DIRECTORY = Path(__file__).resolve().parents[2] / "third_party" / "veli"


def _load_upstream() -> ModuleType:
    if not (UPSTREAM_DIRECTORY / "model" / "Veli.py").is_file():
        raise FileNotFoundError(
            "VELI submodule is missing; run git submodule update --init third_party/veli"
        )
    # Upstream uses absolute imports from generic model and utils packages.
    # Scope the search path to loading, and reject conflicting loaded packages.
    for name in ("model", "utils"):
        existing = sys.modules.get(name)
        if existing is not None:
            source = getattr(existing, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(
                UPSTREAM_DIRECTORY
            ):
                raise ImportError(
                    f"Loaded package {name!r} conflicts with upstream Veli"
                )
    sys.path.insert(0, str(UPSTREAM_DIRECTORY))
    try:
        return importlib.import_module("model.Veli")
    finally:
        sys.path.remove(str(UPSTREAM_DIRECTORY))


class VeliBaseline(_BaselineBase):
    """Use upstream model, loss, initialization, and training without rewriting them.

    Defaults match Veli_train_from_scratch.json. NCAM supplies its training split
    in physical sensor units, rather than upstream AQ-SDR CSVs. The point readout
    is the median of the Gaussian mixture over latent draws of corrected sensor
    averages. Variance is an
    NCAM interface extension using conditional independence given z and latent
    sampling. Both posterior outputs are in physical observation units.
    """

    posterior_in_obs_space = True

    def __init__(
        self,
        n_sensors: int,
        latent_z: int = 1,
        hidden_dim: int = 32,
        epochs: int = 30,
        lr: float = 1.0e-6,
        batch_size: int = 64,
        predict_samples: int = 1,
        beta_z: float = 10.0,
        beta_y: float = 0.1,
        recon_factor: float = 1.0,
        warmup: int = 0,
        seed: int = 1999,
        anchor_idx: int = 0,
        obs_transform: str = "none",
        optimizer_step_size: int = 10,
        optimizer_gamma: float = 0.5,
        reorder_probability: float = 0.5,
        clip_gradient: bool = True,
        no_relu: bool = False,
    ) -> None:
        super().__init__(n_sensors)
        if min(n_sensors, latent_z, epochs, batch_size, predict_samples) < 1:
            raise ValueError(
                "Veli dimensions, epochs, and sample counts must be positive"
            )
        if hidden_dim < 2 or optimizer_step_size < 1:
            raise ValueError("Veli hidden_dim must be >= 2 and scheduler step >= 1")
        if not 0.0 <= reorder_probability <= 1.0:
            raise ValueError("Veli reorder_probability must be in [0, 1]")
        self._upstream = _load_upstream()
        self._latent_z = latent_z
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._predict_samples = predict_samples
        self._beta_z = beta_z
        self._beta_y = beta_y
        self._recon_factor = recon_factor
        self._seed = seed
        self._obs_transform = obs_transform
        self._optimizer_step_size = optimizer_step_size
        self._optimizer_gamma = optimizer_gamma
        self._reorder_probability = reorder_probability
        self._clip_gradient = clip_gradient
        self._no_relu = no_relu
        # Compatibility argument; upstream mean evaluation uses all sensors.
        self._anchor_idx = anchor_idx
        self._model: torch.nn.Module | None = None

    def _physical_readings(self, B: torch.Tensor) -> torch.Tensor:
        if B.ndim != 2 or B.shape[1] != self.n_sensors:
            raise ValueError("Veli expects a row-by-sensor input matrix")
        values = B.exp() if self._obs_transform == "log" else B
        if not torch.isfinite(values).all():
            raise ValueError("The NCAM Veli adapter requires finite sensor readings")
        return values

    def fit(self, B_train: torch.Tensor) -> None:
        readings = self._physical_readings(B_train)
        if len(readings) == 0:
            raise ValueError("Cannot fit Veli on an empty training split")
        data = torch.stack((readings, torch.ones_like(readings)), dim=1)
        cuda_devices = list(range(torch.cuda.device_count()))
        random_state = random.getstate()
        previous_device = self._upstream.device
        try:
            # Upstream train_vae uses this module-level device and Python RNG.
            self._upstream.device = B_train.device
            random.seed(self._seed)
            with (
                torch.random.fork_rng(devices=cuda_devices),
                tempfile.TemporaryDirectory() as logs,
            ):
                torch.manual_seed(self._seed)
                model = self._upstream.Veli(
                    self.n_sensors, self._latent_z, self._hidden_dim, self._no_relu
                ).to(device=B_train.device, dtype=B_train.dtype)
                model.apply(self._upstream.init_weights)
                optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self._optimizer_step_size,
                    gamma=self._optimizer_gamma,
                )
                # train_veli.py actually passes train_loader (shuffle=False),
                # even though it also constructs an unused shuffled epoch_data.
                loader = DataLoader(data, batch_size=self._batch_size, shuffle=False)
                for epoch in range(1, self._epochs + 1):
                    loss, *_ = self._upstream.train_vae(
                        model,
                        loader,
                        optimizer,
                        epoch,
                        beta_z=self._beta_z,
                        beta_y=self._beta_y,
                        reconstruction_factor=self._recon_factor,
                        reorder_probability=self._reorder_probability,
                        loss_type="huber",
                        print_loss=False,
                        clip_gradient=self._clip_gradient,
                        train_log_path=str(Path(logs) / "train.log"),
                    )
                    if not torch.isfinite(torch.tensor(loss)):
                        raise FloatingPointError("Upstream Veli loss became non-finite")
                    scheduler.step()
                model.eval()
                self._model = model
        finally:
            random.setstate(random_state)
            self._upstream.device = previous_device

    def predict_posterior(
        self, x: torch.Tensor, B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._model is None:
            raise RuntimeError("Fit Veli before predicting")
        readings = self._physical_readings(B)
        if len(readings) == 0:
            empty = B.new_empty((0, 1))
            return empty, empty.clone()
        columns = [f"sensor{i}" for i in range(self.n_sensors)]
        frame = pd.DataFrame(readings.detach().cpu().numpy(), columns=columns)
        frame.index.name = "time"
        data = torch.stack((readings, torch.ones_like(readings)), dim=1)
        loader = DataLoader(data, batch_size=self._batch_size, shuffle=False)
        draws, conditional_vars = [], []
        cuda_devices = list(range(torch.cuda.device_count()))
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self._seed)
            for _ in range(self._predict_samples):
                output = self._upstream.infer_to_dataframe(
                    self._model, frame, loader, return_mu_var=True, mu_var_y=True
                )
                corrected = torch.as_tensor(
                    output[columns].to_numpy(), device=B.device, dtype=B.dtype
                )
                y_var = torch.as_tensor(
                    output[[f"var{i}" for i in range(self.n_sensors)]].to_numpy(),
                    device=B.device,
                    dtype=B.dtype,
                )
                draws.append(corrected.mean(dim=1, keepdim=True))
                conditional_vars.append(
                    y_var.sum(dim=1, keepdim=True) / self.n_sensors**2
                )
        samples = torch.stack(draws)
        conditional_var = torch.stack(conditional_vars)
        var = conditional_var.mean(dim=0) + samples.var(dim=0, unbiased=False)
        if len(samples) == 1:
            median = samples[0]
        else:
            std = conditional_var.clamp_min(EPS).sqrt()
            lower = (samples - 8 * std).amin(dim=0)
            upper = (samples + 8 * std).amax(dim=0)
            for _ in range(48):
                midpoint = (lower + upper) / 2
                cdf = (
                    0.5
                    * (
                        1
                        + torch.erf((midpoint.unsqueeze(0) - samples) / (std * 2**0.5))
                    )
                ).mean(dim=0)
                lower = torch.where(cdf < 0.5, midpoint, lower)
                upper = torch.where(cdf >= 0.5, midpoint, upper)
            median = (lower + upper) / 2
        if not torch.isfinite(median).all() or not torch.isfinite(var).all():
            raise FloatingPointError("Upstream Veli predictions became non-finite")
        return median, var.clamp(min=EPS)
