import math

import torch

LOG_2PI = math.log(2 * math.pi)

# Variance clamp bounds (log-space).
# exp(-5) ≈ 0.0067 (std ≈ 0.082)
# exp(4)  ≈ 54.6   (std ≈ 7.39)
LOG_VAR_MIN = -5.0
LOG_VAR_MAX = 4.0


def soft_clamp(x, lo, hi, sharpness=5.0, *, method="softplus"):
    """Bound log variance with an approximately identity map in the interior.

    Two stable forms of the softplus difference avoid cancellation for extreme inputs.
    """
    mid = (lo + hi) / 2.0
    if method != "softplus" or not lo < hi or sharpness <= 0:
        raise ValueError("Invalid variance transform or bounds")
    lower_form = lo + torch.nn.functional.softplus(x - lo, beta=sharpness)
    lower_form = lower_form - torch.nn.functional.softplus(x - hi, beta=sharpness)
    upper_form = hi + torch.nn.functional.softplus(lo - x, beta=sharpness)
    upper_form = upper_form - torch.nn.functional.softplus(hi - x, beta=sharpness)
    return torch.where(x < mid, lower_form, upper_form)


def nll_marginal_sensors(
    b,
    mu0,
    log_var0,
    log_varj,
    multiplicative_bias,
    additive_bias,
    epsilon=1e-8,
    *,
    variance_transform="softplus",
):
    """Negative log-likelihood of the marginal distribution of sensors B,
    marginalizing out the latent variable Theta.
    """
    var0 = (
        torch.exp(
            soft_clamp(log_var0, LOG_VAR_MIN, LOG_VAR_MAX, method=variance_transform)
        )
        + epsilon
    )
    varj = (
        torch.exp(
            soft_clamp(log_varj, LOG_VAR_MIN, LOG_VAR_MAX, method=variance_transform)
        )
        + epsilon
    )

    prec0 = 1.0 / var0
    precj = 1.0 / varj

    marginal_mean = additive_bias + multiplicative_bias * mu0
    error = b - marginal_mean

    sum_sensor_precision = (multiplicative_bias.pow(2) * precj).sum(dim=1, keepdim=True)
    total_precision = prec0 + sum_sensor_precision

    error_projection = (error * multiplicative_bias * precj).sum(dim=1, keepdim=True)
    # The same rank-one Gaussian quadratic, evaluated as a sum of squares.
    # Direct subtraction of the two large terms can lose all float32 accuracy.
    posterior_shift = error_projection / total_precision
    centered = error - multiplicative_bias * posterior_shift
    mahalanobis_dist = (centered.square() * precj).sum(dim=1, keepdim=True)
    mahalanobis_dist = mahalanobis_dist + posterior_shift.square() * prec0

    log_det_diag = torch.log(varj).sum(dim=1, keepdim=True)
    log_det_rank1 = torch.log1p(var0 * sum_sensor_precision)
    log_det = log_det_diag + log_det_rank1

    num_sensors = b.shape[1]
    nll = 0.5 * (mahalanobis_dist + log_det + num_sensors * LOG_2PI)

    return nll.mean()


def variance_regularization(log_var0, log_varj):
    """L2 penalty pulling log-variances toward zero (paper R_var)."""
    penalty_0 = log_var0.pow(2).mean()
    penalty_j = log_varj.pow(2).mean()
    return penalty_0 + penalty_j


def compute_regularization(log_var0, log_varj, var_reg_weight):
    """Compute the variance regularization term R_var.

    Returns:
        var_reg_raw, var_reg_weighted
    """
    device = log_var0.device
    dtype = log_var0.dtype

    var_reg_raw = torch.zeros((), device=device, dtype=dtype)
    if var_reg_weight > 0.0:
        var_reg_raw = variance_regularization(log_var0, log_varj)
    var_reg_weighted = var_reg_weight * var_reg_raw

    return var_reg_raw, var_reg_weighted
