from decimal import ROUND_CEILING, Decimal

import numpy as np
import torch


def conformal_rank(permutations: int, alpha: float) -> int:
    """Candidate-inclusive rank without floating-point jumps at integer boundaries."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    return int(
        (
            Decimal(int(permutations)) * (Decimal(1) - Decimal(str(alpha)))
        ).to_integral_value(rounding=ROUND_CEILING)
    )


def conformal_order_statistic(scores, alpha: float) -> float:
    """Exact split rank, including the unobserved candidate as an extra score.

    An unavailable rank requires an unbounded prediction set. Interpolated
    quantiles and clipping the quantile level change this finite-sample rule.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("calibration scores must be finite")
    rank = conformal_rank(values.size + 1, alpha)
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])


def temporal_calibration_metadata(count, *, alpha, block_size, scheme="nob"):
    """Describe the permutation orbit for one candidate timestamp."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if (
        isinstance(block_size, (bool, np.bool_))
        or not isinstance(block_size, (int, np.integer))
        or block_size < 1
    ):
        raise ValueError("block_size must be a positive integer")
    if scheme not in {"nob", "cyclic"}:
        raise ValueError("scheme must be nob or cyclic")
    if scheme == "cyclic" and block_size != 1:
        raise ValueError("All cyclic shifts require block_size=1")
    permutations = count + 1 if scheme == "cyclic" else (count + 1) // block_size
    retained = max(0, permutations * block_size - 1)
    return {
        "cal_permutation_scores": max(0, permutations - 1),
        "permutations": permutations,
        "threshold_rank": conformal_rank(permutations, alpha),
        "cal_timestamps_discarded": count - retained,
        "cal_block_ends": max(0, permutations - 1) if scheme == "nob" else None,
    }


def split_conformal(
    X_cal,
    B_cal,
    Y_cal,
    ncam_model,
    alpha=0.1,
    sigma_y=0.0,
    normalize_scores: bool = True,
    epsilon=1e-8,
):
    """
    Standard Split Conformal Prediction using Real Labels (Y_cal).
    Guarantees coverage if (X, B, Y) are i.i.d.

    Args:
        X_cal: Calibration covariates (torch.Tensor)
        B_cal: Calibration sensors (torch.Tensor)
        Y_cal: True target values (torch.Tensor)
        ncam_model: Trained model
        alpha: Error rate (0.1 = 90% coverage)
    Returns:
        tau: Calibrated normalized threshold q for locally adaptive intervals
    """

    with torch.no_grad():
        mu_post, var_post = ncam_model.predict_posterior(X_cal, B_cal)
        mu_post = mu_post.squeeze(-1)  # (n,)
        var_post = torch.clamp(var_post.squeeze(-1), min=0.0)
        pred_std = torch.sqrt(torch.clamp(var_post + float(sigma_y) ** 2, min=epsilon))

        target = torch.as_tensor(Y_cal, device=mu_post.device).reshape(-1)
        if target.shape != mu_post.shape:
            raise ValueError("Y_cal must contain one target per calibration row")
        residuals = torch.abs(target - mu_post)
        scores = residuals / pred_std if normalize_scores else residuals

    return conformal_order_statistic(scores.detach().cpu().numpy(), alpha)


def temporal_block_conformal(
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    alpha: float,
    block_size: int,
    scheme: str = "nob",
) -> float:
    """Inductive permutation threshold for one future observed timestamp.

    Each timestamp receives the maximum row score, giving simultaneous
    coverage across locations observed at that time. The augmented sequence
    consists of recent calibration timestamps plus one candidate test time.
    Blocks count observed timestamps, not elapsed clock time. This procedure
    does not give joint coverage across multiple future timestamps.
    ``nob`` shifts by ``block_size`` place block ends at the test position.
    ``cyclic`` uses every shift, with ``block_size=1``, so every calibration
    timestamp contributes a score. The scores need not be independent;
    time-series validity requires suitable invariance/approximation assumptions.
    The identity contributes the candidate score, so its finite-sample rank
    is computed from the other block ends without observing the test label.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    timestamps = np.asarray(timestamps).reshape(-1)
    temporal_calibration_metadata(0, alpha=alpha, block_size=block_size, scheme=scheme)
    if scores.size != timestamps.size:
        raise ValueError("scores and timestamps must have the same length")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if scores.size == 0:
        return float("inf")

    times, inverse = np.unique(timestamps, return_inverse=True)
    if np.any(np.asarray(timestamps != timestamps)):
        raise ValueError("timestamps must not contain missing values")
    time_scores = np.full(times.size, -np.inf)
    np.maximum.at(time_scores, inverse, scores)
    if scheme == "cyclic":
        return conformal_order_statistic(time_scores, alpha)
    # T must be divisible by b. Keep the most recent complete blocks, with
    # the unobserved test timestamp occupying the final position.
    n_blocks = (times.size + 1) // block_size
    if n_blocks == 0:
        return float("inf")
    recent = time_scores[-(n_blocks * block_size - 1) :]
    shifted_scores = recent[block_size - 1 :: block_size]
    return conformal_order_statistic(shifted_scores, alpha)


def predict_interval(X_test, B_test, ncam_model, tau, sigma_y=0.0, epsilon=1e-8):
    """Generate locally adaptive prediction intervals for test locations."""
    with torch.no_grad():
        mu_post, var_post = ncam_model.predict_posterior(X_test, B_test)  # (n, 1)
        mu_post = mu_post.squeeze(-1)
        var_post = torch.clamp(var_post.squeeze(-1), min=0.0)
        pred_std = torch.sqrt(torch.clamp(var_post + float(sigma_y) ** 2, min=epsilon))

        width = float(tau) * pred_std

        lower = mu_post - width
        upper = mu_post + width

    mu = mu_post.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()
    upper = upper.detach().cpu().numpy()
    return mu, lower, upper
