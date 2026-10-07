"""Mean, inverse-variance, PPCA, and Kalman fusion baselines."""

from __future__ import annotations

import torch
import torch.nn as nn

from ncam.kalman import KalmanFilterBaseline  # noqa: F401

EPS = 1e-8


class _BaselineBase(nn.Module):
    """NCAM-compatible interface for baselines."""

    n_sensors: int

    def __init__(self, n_sensors: int) -> None:
        super().__init__()
        self.n_sensors = n_sensors

    def predict_posterior(
        self, x: torch.Tensor, B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu_post, var_post)`` each of shape ``(n, 1)``."""
        raise NotImplementedError


class SensorMeanBaseline(_BaselineBase):
    r"""Arithmetic mean fusion.

    .. math::
        \mu = \mathrm{mean}(B), \qquad
        \sigma^2 = \mathrm{var}(B) / L

    where :math:`L` is the number of sensors. No fitting required.
    """

    def predict_posterior(self, x, B):
        mu = torch.mean(B, dim=1, keepdim=True)
        var = torch.var(B, dim=1, keepdim=True, unbiased=False) / self.n_sensors
        return mu, torch.clamp(var, min=EPS)


class InverseVarianceWeightedMean(_BaselineBase):
    r"""Precision-weighted mean.

    Per-sensor precisions :math:`w_j = 1/\sigma_j^2` are estimated from
    training residuals around the row-wise median (a robust target proxy):

    .. math::
        \mu = \frac{\sum_j w_j B_j}{\sum_j w_j}, \qquad
        \sigma^2 = \frac{1}{\sum_j w_j}.
    """

    def __init__(self, n_sensors: int) -> None:
        super().__init__(n_sensors)
        self.register_buffer("sensor_precision", torch.ones(n_sensors))

    def fit(self, B_train: torch.Tensor) -> None:
        row_median = torch.median(B_train, dim=1).values.unsqueeze(1)
        residuals = B_train - row_median
        sensor_var = torch.var(residuals, dim=0, unbiased=False).clamp(min=EPS)
        self.sensor_precision = 1.0 / sensor_var

    def predict_posterior(self, x, B):
        w = self.sensor_precision.unsqueeze(0)
        total_prec = w.sum()
        mu = (w * B).sum(dim=1, keepdim=True) / total_prec
        var = (1.0 / total_prec).expand(B.shape[0], 1)
        return mu, var


class PPCABaseline(_BaselineBase):
    r"""Single-factor probabilistic PCA (factor analysis with one factor).

    Observation model per sensor :math:`j`:

    .. math::
        B_j = w_j \theta + \mu_j + \varepsilon_j,
        \quad \varepsilon_j \sim \mathcal{N}(0, \psi_j),
        \quad \theta \sim \mathcal{N}(0, 1).

    Loadings :math:`w_j`, biases :math:`\mu_j`, and noise variances
    :math:`\psi_j` are estimated by EM on training observations. At
    prediction time, the latent posterior is mapped to observation scale
    using the mean loading and mean bias.
    """

    def __init__(self, n_sensors: int, n_em_iter: int = 100) -> None:
        super().__init__(n_sensors)
        self.n_em_iter = n_em_iter
        self.register_buffer("loadings", torch.ones(n_sensors))
        self.register_buffer("means", torch.zeros(n_sensors))
        self.register_buffer("noise_var", torch.ones(n_sensors))

    def fit(self, B_train: torch.Tensor) -> None:
        B_c = B_train.cpu()
        n = B_c.shape[0]
        mu = B_c.mean(dim=0)
        B_c = B_c - mu

        cov = (B_c.T @ B_c) / n
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        w = eigenvectors[:, -1] * torch.sqrt(eigenvalues[-1].clamp(min=EPS))
        psi = torch.diagonal(cov).clone().clamp(min=EPS)
        if w.sum() < 0:
            w = -w  # convention: sensors load positively on the latent

        for _ in range(self.n_em_iter):
            # E-step
            var_theta = 1.0 / (1.0 + (w**2 / psi).sum())
            E_theta = var_theta * (B_c * (w / psi).unsqueeze(0)).sum(dim=1)
            E_theta2 = var_theta + E_theta**2

            # M-step
            w = (B_c * E_theta.unsqueeze(1)).sum(dim=0) / E_theta2.sum()
            residuals = B_c - w.unsqueeze(0) * E_theta.unsqueeze(1)
            psi = ((residuals**2).mean(dim=0) + w**2 * var_theta).clamp(min=EPS)

        device = self.loadings.device
        self.loadings = w.to(device)
        self.means = mu.to(device)
        self.noise_var = psi.to(device)

    def predict_posterior(self, x, B):
        w, mu, psi = self.loadings, self.means, self.noise_var
        B_c = B - mu.unsqueeze(0)

        var_theta = 1.0 / (1.0 + (w**2 / psi).sum())
        mu_theta = var_theta * (B_c * (w / psi).unsqueeze(0)).sum(dim=1, keepdim=True)

        w_mean = w.mean()
        mu_post = w_mean * mu_theta + mu.mean()
        var_post = torch.full_like(
            mu_post, max(float((w_mean**2 * var_theta).item()), EPS)
        )
        return mu_post, var_post


class GaussianFusionBaseline(_BaselineBase):
    r"""Conjugate Gaussian sensor fusion (single-step Bayesian update).

    Prior estimated from training row-means:
    :math:`\theta \sim \mathcal{N}(\mu_{\text{prior}}, \sigma^2_{\text{prior}})`.
    Per-sensor observation model:
    :math:`B_j \mid \theta \sim \mathcal{N}(\theta, R_j)`.
    Posterior is the standard conjugate update:

    .. math::
        \sigma^{-2}_{\text{post}} =
            \sigma^{-2}_{\text{prior}} + \sum_j R_j^{-1},
        \qquad
        \mu_{\text{post}} = \sigma^2_{\text{post}}
            \Bigl(\mu_{\text{prior}} \sigma^{-2}_{\text{prior}}
            + \sum_j B_j / R_j \Bigr).
    """

    def __init__(self, n_sensors: int) -> None:
        super().__init__(n_sensors)
        self.register_buffer("prior_mean", torch.tensor(0.0))
        self.register_buffer("prior_var", torch.tensor(1.0))
        self.register_buffer("obs_noise", torch.ones(n_sensors))

    def fit(self, B_train: torch.Tensor) -> None:
        row_means = B_train.mean(dim=1)
        self.prior_mean = row_means.mean()
        self.prior_var = row_means.var(unbiased=False).clamp(min=EPS)

        residuals = B_train - B_train.mean(dim=1, keepdim=True)
        self.obs_noise = residuals.var(dim=0, unbiased=False).clamp(min=EPS)

    def predict_posterior(self, x, B):
        prior_prec = 1.0 / self.prior_var
        obs_prec = 1.0 / self.obs_noise

        total_prec = prior_prec + obs_prec.sum()
        var_scalar = float((1.0 / total_prec).item())

        prior_term = self.prior_mean * prior_prec
        obs_term = (B * obs_prec.unsqueeze(0)).sum(dim=1, keepdim=True)
        mu_post = (prior_term + obs_term) / total_prec
        var_post = torch.full_like(mu_post, max(var_scalar, EPS))
        return mu_post, var_post
