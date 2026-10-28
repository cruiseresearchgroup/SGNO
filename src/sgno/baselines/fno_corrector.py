"""Matched FNO correction network with the same outer propagation and processing."""

from __future__ import annotations
import torch
from torch import nn
from sgno.baselines.fno_1d import ApeBenchFNO1d
from sgno.baselines.fno_nd import ApeBenchFNOND
from sgno._affine import LinearResidualSGNO
from sgno._spec import SPATIAL_SIZES

METHOD = "sgno_affine_fno_shell_v1"
VERSION = "affine-modal-propagator-shell-matched-fno-increment-v1-20260919"
FNO_CONFIG = {
    1: dict(width=9, modes=26, n_blocks=7),
    2: dict(width=4, modes=10, n_blocks=9),
    3: dict(width=7, modes=6, n_blocks=2),
}


def _build_fno(ndim: int):
    cfg = FNO_CONFIG[ndim]
    n = SPATIAL_SIZES[ndim]
    if ndim == 1:
        return ApeBenchFNO1d(
            num_channels=1,
            num_points=n,
            width=cfg["width"],
            modes=cfg["modes"],
            n_blocks=cfg["n_blocks"],
        )
    return ApeBenchFNOND(
        spatial_shape=(n,) * ndim,
        num_channels=1,
        width=cfg["width"],
        modes=cfg["modes"],
        n_blocks=cfg["n_blocks"],
        activation="gelu",
        outer_residual=False,
    )


class AffineFNOShell(LinearResidualSGNO):
    """Keep SGNO's physical shell and replace only its learned increment."""

    def __init__(self, residual: nn.Module, ndim: int):
        super().__init__(residual, ndim)
        self.dc_amplitude = nn.Parameter(torch.tensor(0.01))
        self.axis_amplitude = nn.Parameter(torch.full((ndim,), 0.1))
        n = SPATIAL_SIZES[ndim]
        normalized = [2.0 * torch.fft.fftfreq(n) for _ in range(ndim - 1)]
        normalized.append(2.0 * torch.fft.rfftfreq(n))
        integer = [torch.fft.fftfreq(n, d=1.0 / n) for _ in range(ndim - 1)]
        integer.append(torch.fft.rfftfreq(n, d=1.0 / n))
        normalized_mesh = torch.meshgrid(*normalized, indexing="ij")
        integer_mesh = torch.meshgrid(*integer, indexing="ij")
        self.register_buffer(
            "normalized_frequency_squared",
            torch.stack([value.square() for value in normalized_mesh]),
            persistent=False,
        )
        cutoff = FNO_CONFIG[ndim]["modes"]
        high_mask = torch.zeros_like(integer_mesh[0], dtype=torch.bool)
        for value in integer_mesh:
            high_mask |= value.abs() >= cutoff
        self.register_buffer("high_mask", high_mask, persistent=False)
        self._eval_zero_response = None

    @torch.no_grad()
    def initialize_from_train(self, data):
        receipt = LinearResidualSGNO.initialize_from_train(self, data)
        self.residual.project.weight.zero_()
        self.residual.project.bias.zero_()
        self.residual_weight.fill_(1.0)
        receipt.update(
            increment="Mode/depth-matched FNO topology in the current SGNO shell",
            increment_readout_initial="zero weight and bias",
            residual_gate_initial=1.0,
            subsequent_training="Affine propagator, output filter, and FNO increment trained jointly",
        )
        return receipt

    def train(self, mode: bool = True):
        self._eval_zero_response = None
        return super().train(mode)

    def _apply(self, fn, recurse: bool = True):
        self._eval_zero_response = None
        return super()._apply(fn, recurse=recurse)

    def exponent(self):
        shape = (self.ndim,) + (1,) * self.ndim
        return -self.dc_amplitude.square() - (
            self.axis_amplitude.square().reshape(shape)
            * self.normalized_frequency_squared
        ).sum(0)

    def output_filter(self, value):
        dims = tuple(range(-self.ndim, 0))
        spectrum = torch.fft.rfftn(value.float(), dim=dims)
        correction = torch.fft.irfftn(
            torch.expm1(self.exponent()) * spectrum, s=self.spatial_shape, dim=dims
        )
        return value + correction.to(value.dtype)

    def highpass(self, value):
        dims = tuple(range(-self.ndim, 0))
        spectrum = torch.fft.rfftn(value.float(), dim=dims)
        return torch.fft.irfftn(
            spectrum * self.high_mask, s=self.spatial_shape, dim=dims
        ).to(value.dtype)

    def zero_response(self, template):
        if not self.training and (not torch.is_grad_enabled()):
            if self._eval_zero_response is None:
                zero = torch.zeros_like(template[:1])
                self._eval_zero_response = self.highpass(
                    self.output_filter(self.residual(zero))
                ).detach()
            return self._eval_zero_response
        zero = torch.zeros_like(template[:1])
        return self.highpass(self.output_filter(self.residual(zero)))

    def forward(self, u):
        axes = tuple(range(2, u.ndim))
        mean = u.mean(axes, keepdim=True)
        centered = u - mean
        increment = self.output_filter(self.residual(centered)) - self.zero_response(
            centered
        )
        prediction = self.linear(centered) + self.residual_weight * increment
        return prediction - prediction.mean(axes, keepdim=True) + mean

    def learned_state_propagation(self):
        cfg = FNO_CONFIG[self.ndim]
        return dict(
            scope="Mean-preserving fitted affine propagation plus a mode/depth-matched FNO increment with output filtering and high-frequency zero-response subtraction",
            residual_weight=float(self.residual_weight.detach()),
            state_multiplier_max_gain=float(self.multiplier().detach().abs().max()),
            state_multiplier_min_gain=float(self.multiplier().detach().abs().min()),
            forcing_l2=float(torch.linalg.vector_norm(self.forcing.detach())),
            fno_width=cfg["width"],
            fno_modes=cfg["modes"],
            fno_blocks=cfg["n_blocks"],
            output_filter_dc=float(self.dc_amplitude.detach()),
            output_filter_axis=self.axis_amplitude.detach().cpu().tolist(),
        )


def build_model(ndim: int, seed: int = 0):
    if ndim not in (1, 2, 3) or seed != 0:
        raise ValueError("fixed all-fourteen-task seed-zero screen")
    torch.manual_seed(seed)
    return AffineFNOShell(_build_fno(ndim), ndim)


def model_config(ndim: int):
    model = build_model(ndim, seed=0)
    cfg = FNO_CONFIG[ndim]
    elements = sum((parameter.numel() for parameter in model.parameters()))
    real = sum(
        (
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
    )
    return dict(
        version=VERSION,
        ndim=ndim,
        width=cfg["width"],
        modes=cfg["modes"],
        n_blocks=cfg["n_blocks"],
        tensor_entries=elements,
        real_parameters=real,
        outer_propagator="training-pair fitted affine Fourier map",
        mean_preservation=True,
        output_filter=True,
        high_frequency_zero_response_subtraction=True,
        zero_initialized_increment_readout=True,
        task_specific_switches=False,
    )


def model_spec(model):
    cfg = model_config(model.ndim)
    elements = sum((parameter.numel() for parameter in model.parameters()))
    real = sum(
        (
            parameter.numel() * (2 if parameter.is_complex() else 1)
            for parameter in model.parameters()
        )
    )
    assert (elements, real) == (cfg["tensor_entries"], cfg["real_parameters"])
    return dict(
        method=METHOD,
        version=VERSION,
        ndim=model.ndim,
        config=cfg,
        parameter_counts=dict(tensor_entries=elements, real_parameters=real),
        prediction="Input mean plus fitted affine modal propagation and a filtered, zero-response-corrected FNO increment of the centered state",
    )
