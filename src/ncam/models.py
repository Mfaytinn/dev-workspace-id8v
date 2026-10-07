import torch
import torch.nn as nn

from ncam.loss import LOG_VAR_MAX, LOG_VAR_MIN, soft_clamp

DEFAULT_OFFSET_PARAMETERIZATION = "mean_residual"


def _get_final_linear(module_with_net) -> nn.Linear:
    return next(
        layer
        for layer in reversed(list(module_with_net.modules()))
        if isinstance(layer, nn.Linear)
    )


def _variance_from_log(
    log_var,
    min_value=LOG_VAR_MIN,
    max_value=LOG_VAR_MAX,
    epsilon=1e-8,
    *,
    method="softplus",
):
    return torch.exp(soft_clamp(log_var, min_value, max_value, method=method)) + epsilon


def _kaiming_init_linear(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _kaiming_init_linear_layers(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            _kaiming_init_linear(layer)


def _init_variance_head(mlp_module, output_indices, target_log_var=0.0):
    """Initialize specific output channels of an MLP to produce target_log_var.

    Sets the final linear layer's weights to near-zero and bias to target_log_var
    for the given output indices, so variance networks start inside the clamp
    range instead of overshooting into the dead zone.
    """
    final = _get_final_linear(mlp_module)
    with torch.no_grad():
        for idx in output_indices:
            final.weight[idx].zero_()
            final.bias[idx] = target_log_var


def _predict_posterior_from_params(
    B,
    mu_0,
    log_var_0,
    log_var_j,
    multiplicative_bias,
    additive_bias,
    *,
    variance_transform="softplus",
):
    var_0 = _variance_from_log(log_var_0, method=variance_transform)
    var_j = _variance_from_log(log_var_j, method=variance_transform)

    prec_0 = 1.0 / var_0
    prec_j = 1.0 / var_j

    weighted_prec = (multiplicative_bias**2) * prec_j
    posterior_prec = prec_0 + torch.sum(weighted_prec, dim=1, keepdim=True)
    var_post = 1.0 / posterior_prec

    prior_contribution = mu_0 * prec_0
    residuals = B - additive_bias
    sensor_contribution = prec_j * multiplicative_bias * residuals
    mu_post = var_post * (
        prior_contribution + torch.sum(sensor_contribution, dim=1, keepdim=True)
    )
    return mu_post, var_post


class _NCAMBase(nn.Module):
    """Shared inference/accessor methods for NCAM model variants."""

    def _prior_parameters(self, x):
        prior = self.prior_net(x)
        return prior[:, 0:1], prior[:, 1:2]

    def predict_posterior(self, x, B):
        return _predict_posterior_from_params(
            B, *self.forward(x), variance_transform=self.variance_transform
        )

    def get_sensor_variances(self, x):
        return _variance_from_log(self.forward(x)[2], method=self.variance_transform)

    def get_prior_variance(self, x):
        return _variance_from_log(self.forward(x)[1], method=self.variance_transform)


class MLP(nn.Module):
    """Three GELU hidden layers used by every paper model head."""

    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        layers = []
        for index in range(3):
            layers.extend(
                (nn.Linear(in_dim if index == 0 else hidden_dim, hidden_dim), nn.GELU())
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        _kaiming_init_linear_layers(self.net)

    def forward(self, x):
        return self.net(x)


class NCAMNoBias(_NCAMBase):
    """NCAM without learned calibration.

    This model fixes gains=1 and offsets=0, and learns only prior and
    sensor variances for fusion weighting.
    """

    def __init__(
        self,
        in_dim,
        n_sensors,
        hidden_dim=128,
        prior_hidden_dim=None,
        rel_hidden_dim=None,
        bias_hidden_dim=None,
        variance_transform="softplus",
    ):
        super().__init__()
        if variance_transform != "softplus":
            raise ValueError("NCAM uses the softplus variance transform")
        self.feature_architecture = "dense"
        self.prior_mean_parameterization = "location"
        self.variance_transform = variance_transform
        self.n_sensors = n_sensors
        h_prior = prior_hidden_dim if prior_hidden_dim is not None else hidden_dim
        self.mlp_depth = 3
        self.mlp_activation = "gelu"
        self.prior_net = MLP(
            in_dim,
            2,
            h_prior,
        )

        h_rel = rel_hidden_dim if rel_hidden_dim is not None else hidden_dim
        self.rel_net = MLP(
            in_dim,
            n_sensors,
            h_rel,
        )

        # Variance heads start at log_var=0 (variance=1) instead of random
        # values that overshoot the clamp and get stuck in the dead zone.
        _init_variance_head(self.prior_net, [1], target_log_var=0.0)
        _init_variance_head(self.rel_net, list(range(n_sensors)), target_log_var=0.0)

    def forward(self, x):
        mu_0, log_var_0 = self._prior_parameters(x)

        log_var_j = self.rel_net(x)

        multiplicative_bias = torch.ones_like(log_var_j)
        additive_bias = torch.zeros_like(log_var_j)

        return mu_0, log_var_0, log_var_j, multiplicative_bias, additive_bias


class _NCAMCalibrated(_NCAMBase):
    """Contextual NCAM with optional sensor anchoring and learned calibration."""

    _requires_anchor = False

    def __init__(
        self,
        in_dim,
        n_sensors,
        anchor_idx=0,
        hidden_dim=128,
        prior_hidden_dim=None,
        rel_hidden_dim=None,
        bias_hidden_dim=None,
        variance_transform="softplus",
        offset_parameterization=DEFAULT_OFFSET_PARAMETERIZATION,
    ):
        super().__init__()
        if variance_transform != "softplus":
            raise ValueError("NCAM uses the softplus variance transform")
        self.feature_architecture = "dense"
        self.prior_mean_parameterization = "location"
        if offset_parameterization != DEFAULT_OFFSET_PARAMETERIZATION:
            raise ValueError("Unsupported offset parameterization")
        self.offset_parameterization = offset_parameterization
        self.gain_parameterization = "direct"
        self.variance_transform = variance_transform
        self.prior_mode = "contextual"
        self.gain_mode = "learned"
        self.n_sensors = n_sensors
        if n_sensors < 1:
            raise ValueError("NCAM requires at least one sensor")
        if anchor_idx is None:
            if self._requires_anchor:
                raise ValueError("Anchored NCAM requires an anchor index")
        else:
            if not isinstance(anchor_idx, int) or not 0 <= anchor_idx < n_sensors:
                raise ValueError("Anchor index must identify an input sensor")
            self.anchor_idx = anchor_idx
        self._calibration_indices = tuple(
            index for index in range(n_sensors) if index != anchor_idx
        )

        h_prior = prior_hidden_dim if prior_hidden_dim is not None else hidden_dim
        self.mlp_depth = 3
        self.mlp_activation = "gelu"
        self.prior_net = MLP(
            in_dim,
            2,
            h_prior,
        )

        h_rel = rel_hidden_dim if rel_hidden_dim is not None else hidden_dim
        self.rel_net = MLP(
            in_dim,
            n_sensors,
            h_rel,
        )

        h_bias = bias_hidden_dim if bias_hidden_dim is not None else hidden_dim
        bias_outputs = 2 * len(self._calibration_indices)
        self.bias_net = MLP(
            in_dim,
            bias_outputs,
            h_bias,
        )
        self._init_bias_net_output()

        _init_variance_head(self.prior_net, [1], target_log_var=0.0)
        _init_variance_head(self.rel_net, list(range(n_sensors)), target_log_var=0.0)

    def _init_bias_net_output(self):
        n_learnable = len(self._calibration_indices)
        final_layer = _get_final_linear(self.bias_net)
        with torch.no_grad():
            # softplus(0.5414) ≈ 1.0 → multiplicative bias starts near 1
            final_layer.bias[:n_learnable].fill_(0.5414)
            final_layer.weight[:n_learnable].zero_()
            final_layer.bias[n_learnable:].zero_()
            final_layer.weight[n_learnable:].zero_()

    def _expand_bias_with_anchor(self, bias_learnable, anchor_value):
        if not hasattr(self, "anchor_idx"):
            return bias_learnable
        batch_size = bias_learnable.shape[0]
        device = bias_learnable.device
        dtype = bias_learnable.dtype

        anchor_col = torch.full(
            (batch_size, 1), anchor_value, device=device, dtype=dtype
        )

        if self.anchor_idx == 0:
            return torch.cat([anchor_col, bias_learnable], dim=1)
        if self.anchor_idx == self.n_sensors - 1:
            return torch.cat([bias_learnable, anchor_col], dim=1)
        left = bias_learnable[:, : self.anchor_idx]
        right = bias_learnable[:, self.anchor_idx :]
        return torch.cat([left, anchor_col, right], dim=1)

    def forward(self, x):
        mu_0, log_var_0 = self._prior_parameters(x)

        log_var_j = self.rel_net(x)

        bias = self.bias_net(x)
        n_learnable = len(self._calibration_indices)
        mult_learnable = bias[:, :n_learnable]
        add_learnable = bias[:, n_learnable:]

        mult_learnable = torch.nn.functional.softplus(mult_learnable)
        mult_learnable = torch.clamp(mult_learnable, min=0.2, max=5.0)
        # Separate the marginal mean from covariance gains. The nominal
        # gain matches the existing initialization, preserving initial c=0.
        nominal_gain = torch.nn.functional.softplus(
            torch.full_like(mult_learnable, 0.5414)
        )
        add_learnable = add_learnable + (nominal_gain - mult_learnable) * mu_0
        add_learnable = torch.clamp(add_learnable, min=-20.0, max=20.0)

        multiplicative_bias = self._expand_bias_with_anchor(
            mult_learnable, anchor_value=1.0
        )
        additive_bias = self._expand_bias_with_anchor(add_learnable, anchor_value=0.0)

        return mu_0, log_var_0, log_var_j, multiplicative_bias, additive_bias


class NCAMAnchored(_NCAMCalibrated):
    """Mean-residual NCAM with sensor anchoring for identifiability.

    Fixing one sensor's gain to 1 and offset to 0 removes the latent affine
    ambiguity. All other sensors retain learned contextual calibration.
    """

    _requires_anchor = True


class NCAMUnanchored(_NCAMCalibrated):
    """Identifiability ablation with learned gains and offsets for every sensor.

    The contextual prior, variance heads, calibration parameterizations, and
    conjugate inference match anchored NCAM, without fixing a sensor's scale
    or location. The latent scale and location are not identified by the
    marginal sensor likelihood alone.
    """

    def __init__(
        self,
        in_dim,
        n_sensors,
        hidden_dim=128,
        prior_hidden_dim=None,
        rel_hidden_dim=None,
        bias_hidden_dim=None,
        variance_transform="softplus",
        offset_parameterization=DEFAULT_OFFSET_PARAMETERIZATION,
    ):
        super().__init__(
            in_dim=in_dim,
            n_sensors=n_sensors,
            anchor_idx=None,
            hidden_dim=hidden_dim,
            prior_hidden_dim=prior_hidden_dim,
            rel_hidden_dim=rel_hidden_dim,
            bias_hidden_dim=bias_hidden_dim,
            variance_transform=variance_transform,
            offset_parameterization=offset_parameterization,
        )
