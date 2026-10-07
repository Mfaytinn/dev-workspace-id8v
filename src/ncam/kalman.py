"""Training-only EM and causal filtering for a scalar local-level model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch import nn

VAR_FLOOR = 1e-8
HOUR_NS = 3_600_000_000_000
KALMAN_PROTOCOL = "local_level_train_mle_v1"


class KalmanFilterBaseline(nn.Module):
    r"""Independent location states with shared sensor-noise parameters.

    In the experiment's observation space (log PM for the real datasets):

    .. math::
        z_t = z_{t-1} + w_t,\quad w_t\sim N(0,Q\Delta t),
        B_{tj} = z_t + v_{tj},\quad v_{tj}\sim N(0,R_j).

    Time is measured in hours. Gains are one, offsets zero, and observation
    noise is diagonal. Q and R are fitted by EM using training sensors only,
    with direct marginal-likelihood refinement if EM hits its iteration limit;
    the initial distribution is the empirical training row-mean distribution.
    Smoothing is confined to training EM. Evaluation returns filtered states,
    conditioning on current and earlier sensors, never future observations.

    Each prediction call restarts from the fitted training-end states. Between
    training and test we propagate uncertainty across elapsed time without
    assimilating validation, calibration, or gap rows. This preserves the
    comparison's training/test input restrictions. An unseen location starts
    from the training initial distribution. Rows may arrive in arbitrary order;
    they are filtered by location and timestamp and returned in original order.
    """

    def __init__(
        self, n_sensors: int, n_em_iter: int = 100, em_tolerance: float = 1e-6
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.n_em_iter = n_em_iter
        self.em_tolerance = em_tolerance
        self.register_buffer("prior_mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("prior_var", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("process_noise", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("obs_noise", torch.ones(n_sensors, dtype=torch.float64))
        self.register_buffer("end_means", torch.empty(0, dtype=torch.float64))
        self.register_buffer("end_vars", torch.empty(0, dtype=torch.float64))
        self.register_buffer("end_times", torch.empty(0, dtype=torch.int64))
        self.locations: tuple[str, ...] = ()
        self.fit_diagnostics: dict = {}

    def get_extra_state(self):
        return {"locations": self.locations, "fit_diagnostics": self.fit_diagnostics}

    def set_extra_state(self, state):
        self.locations = tuple(state["locations"])
        self.fit_diagnostics = dict(state["fit_diagnostics"])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The number of trained locations is data-dependent.
        for name in ("end_means", "end_vars", "end_times"):
            key = prefix + name
            if key in state_dict:
                self._buffers[name] = torch.empty_like(
                    state_dict[key], device=self.prior_mean.device
                )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _inputs(self, B: torch.Tensor, keys: pd.DataFrame | None):
        values = B.detach().cpu().double().numpy()
        times = (
            pd.to_datetime(keys.date, utc=True)
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
        )
        locations = keys.location_id.astype(str).to_numpy()
        sequences = []
        for location in np.unique(locations):
            indices = np.flatnonzero(locations == location)
            indices = indices[np.argsort(times[indices], kind="stable")]
            sequences.append((location, indices))
        return values, times, sequences

    @staticmethod
    def _filter(values, times, q, r, mean, variance, previous_time=None):
        n = len(values)
        means, variances, predicted_vars = (np.empty(n) for _ in range(3))
        precision = 1.0 / r
        total_precision = precision.sum()
        effective_var = 1.0 / total_precision
        effective_mean = values @ precision / total_precision
        disagreement = ((values - effective_mean[:, None]) ** 2) @ precision
        logdet_r = np.log(r).sum()
        nll = 0.0
        for t in range(n):
            if previous_time is not None:
                variance += q * float(times[t] - previous_time) / HOUR_NS
            predicted_vars[t] = variance
            innovation_var = variance + effective_var
            nll += 0.5 * (
                len(r) * np.log(2 * np.pi)
                + logdet_r
                + np.log1p(variance * total_precision)
                + disagreement[t]
                + (effective_mean[t] - mean) ** 2 / innovation_var
            )
            gain = variance / innovation_var
            mean += gain * (effective_mean[t] - mean)
            variance = variance * effective_var / innovation_var
            means[t], variances[t] = mean, variance
            previous_time = times[t]
        return means, variances, predicted_vars, float(nll)

    def fit(
        self, B_train: torch.Tensor, train_keys: pd.DataFrame | None = None
    ) -> None:
        values, times, sequences = self._inputs(B_train, train_keys)
        transitions = sum(len(indices) - 1 for _, indices in sequences)
        row_means = values.mean(axis=1)
        prior_mean = float(row_means.mean())
        prior_var = max(float(row_means.var()), VAR_FLOOR)
        # These are initial values only; EM corrects the row-residual heuristic.
        r = np.maximum((values - row_means[:, None]).var(axis=0), VAR_FLOOR)
        if self.n_sensors == 1:
            r[:] = max(prior_var * 0.1, VAR_FLOOR)
        q = max(
            sum(
                np.sum(np.diff(row_means[idx]) ** 2 / (np.diff(times[idx]) / HOUR_NS))
                for _, idx in sequences
            )
            / transitions,
            VAR_FLOOR,
        )
        history = []
        converged = False
        for iteration in range(self.n_em_iter + 1):
            q_sum, r_sum, nll = 0.0, np.zeros(self.n_sensors), 0.0
            for _, idx in sequences:
                b, ts = values[idx], times[idx]
                m, p, predicted_p, sequence_nll = self._filter(
                    b, ts, q, r, prior_mean, prior_var
                )
                nll += sequence_nll
                sm, sp = m.copy(), p.copy()
                for t in range(len(idx) - 2, -1, -1):
                    gain = p[t] / predicted_p[t + 1]
                    sm[t] += gain * (sm[t + 1] - m[t])
                    sp[t] = max(p[t] + gain**2 * (sp[t + 1] - predicted_p[t + 1]), 0.0)
                r_sum += ((b - sm[:, None]) ** 2 + sp[:, None]).sum(axis=0)
                if len(idx) > 1:
                    cross_cov = p[:-1] / predicted_p[1:] * sp[1:]
                    expected_diff2 = np.maximum(
                        np.diff(sm) ** 2 + sp[:-1] + sp[1:] - 2 * cross_cov, 0.0
                    )
                    q_sum += np.sum(expected_diff2 / (np.diff(ts) / HOUR_NS))
            history.append(nll)
            if len(history) > 1 and abs(history[-2] - nll) <= self.em_tolerance * (
                1 + abs(history[-2])
            ):
                converged = True
                break
            if iteration == self.n_em_iter:
                break
            q = max(float(q_sum / transitions), VAR_FLOOR)
            r = np.maximum(r_sum / len(values), VAR_FLOOR)

        optimizer_diagnostics = None
        final_nll = history[-1]
        if not converged:

            def objective(log_variances):
                candidate_q, candidate_r = (
                    np.exp(log_variances[0]),
                    np.exp(log_variances[1:]),
                )
                return sum(
                    self._filter(
                        values[idx],
                        times[idx],
                        candidate_q,
                        candidate_r,
                        prior_mean,
                        prior_var,
                    )[3]
                    for _, idx in sequences
                ) / len(values)

            optimized = minimize(
                objective,
                np.log(np.r_[q, r]),
                method="L-BFGS-B",
                bounds=[(np.log(VAR_FLOOR), None)] * (self.n_sensors + 1),
                options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7, "eps": 1e-5},
            )
            final_nll = float(optimized.fun * len(values))
            q, r = float(np.exp(optimized.x[0])), np.exp(optimized.x[1:])
            optimizer_diagnostics = {
                "method": "L-BFGS-B on log variances",
                "converged": bool(optimized.success),
                "message": str(optimized.message),
                "iterations": int(optimized.nit),
                "evaluations": int(optimized.nfev),
            }

        device = self.prior_mean.device
        for name, value in (
            ("prior_mean", prior_mean),
            ("prior_var", prior_var),
            ("process_noise", q),
            ("obs_noise", r),
        ):
            self._buffers[name] = torch.as_tensor(
                value, dtype=torch.float64, device=device
            )
        ends = []
        for _, idx in sequences:
            m, p, _, _ = self._filter(
                values[idx], times[idx], q, r, prior_mean, prior_var
            )
            ends.append((m[-1], p[-1], times[idx[-1]]))
        self.end_means = torch.tensor(
            [e[0] for e in ends], dtype=torch.float64, device=device
        )
        self.end_vars = torch.tensor(
            [e[1] for e in ends], dtype=torch.float64, device=device
        )
        self.end_times = torch.tensor(
            [e[2] for e in ends], dtype=torch.int64, device=device
        )
        self.locations = tuple(location for location, _ in sequences)
        self.fit_diagnostics = {
            "protocol": KALMAN_PROTOCOL,
            "em_iterations": len(history) - 1,
            "em_converged": converged,
            "training_nll": history,
            "final_training_nll": final_nll,
            "likelihood_refinement": optimizer_diagnostics,
            "fit_converged": converged or bool(optimizer_diagnostics["converged"]),
            "process_variance_per_hour": q,
            "sensor_noise_variances": r.tolist(),
            "training_rows": len(values),
            "training_locations": list(self.locations),
            "reference_values_used": False,
            "measurement_gains": "fixed at one",
            "measurement_offsets": "fixed at zero",
            "measurement_noise": "diagonal; shared across locations",
            "initial_distribution": "training row-mean empirical mean and variance",
            "evaluation": (
                "causal filtering from training end; no val/cal/gap assimilation"
            ),
        }

    def predict_posterior(self, x, B, test_keys: pd.DataFrame | None = None):
        values, times, sequences = self._inputs(B, test_keys)
        mean, variance = np.empty(len(values)), np.empty(len(values))
        q = float(self.process_noise.item())
        r = self.obs_noise.cpu().numpy()
        for location, idx in sequences:
            if location in self.locations:
                group = self.locations.index(location)
                initial = (
                    float(self.end_means[group].item()),
                    float(self.end_vars[group].item()),
                    int(self.end_times[group].item()),
                )
            else:
                initial = (
                    float(self.prior_mean.item()),
                    float(self.prior_var.item()),
                    None,
                )
            m, p, _, _ = self._filter(values[idx], times[idx], q, r, *initial)
            mean[idx], variance[idx] = m, p
        return (
            torch.as_tensor(mean[:, None], dtype=B.dtype, device=B.device),
            torch.as_tensor(variance[:, None], dtype=B.dtype, device=B.device),
        )
