"""Learned Fourier output filter."""

from pathlib import Path
import sys
import torch
from torch import nn
from sgno._spec import (
    build_reference,
    SPATIAL_SIZES,
    ARCHITECTURES,
    BASE_COUNTS,
    CURRENT_PROFILE,
    LEGACY_PROFILE,
)

METHOD = "sgno_complete_update_v1"
VERSION = "complete-update-axis-quadratic-decay-learned-dc-v1-20260917"


def model_config(ndim):
    w, m, l = ARCHITECTURES[ndim]
    elements, real = BASE_COUNTS[ndim]
    return dict(
        width=w,
        modes=m,
        n_blocks=l,
        initial_step=1,
        model_profile=CURRENT_PROFILE if ndim == 2 else LEGACY_PROFILE,
        tensor_entries=elements + ndim + 1,
        real_parameters=real + ndim + 1,
        complete_update=dict(
            version=VERSION,
            formula="G_b[f_original(u)]",
            symbol="exp(-b0^2-sum_j bj^2*(2*kj/Nj)^2)",
            dc_initial=0.01,
            axis_initial=0.1,
            physical_coefficients_supplied=False,
            task_switches=False,
        ),
    )


class CompleteUpdateSGNO(nn.Module):
    def __init__(self, base, ndim):
        super().__init__()
        self.base = base
        self.ndim = ndim
        self.dc_amplitude = nn.Parameter(torch.tensor(0.01))
        self.axis_amplitude = nn.Parameter(torch.full((ndim,), 0.1))
        n = SPATIAL_SIZES[ndim]
        self.spatial_shape = (n,) * ndim
        freq = [2 * torch.fft.fftfreq(n) for _ in range(ndim - 1)] + [
            2 * torch.fft.rfftfreq(n)
        ]
        self.register_buffer(
            "normalized_frequency_squared",
            torch.stack([g.square() for g in torch.meshgrid(*freq, indexing="ij")]),
            persistent=False,
        )

    def exponent(self):
        return -self.dc_amplitude.square() - (
            self.axis_amplitude.square().reshape((self.ndim,) + (1,) * self.ndim)
            * self.normalized_frequency_squared
        ).sum(0)

    def filter(self, x):
        dims = tuple(range(-self.ndim, 0))
        h = torch.fft.rfftn(x.float(), dim=dims)
        correction = torch.fft.irfftn(
            torch.expm1(self.exponent()) * h, s=self.spatial_shape, dim=dims
        )
        return x + correction.to(x.dtype)

    def forward(self, u):
        return self.filter(self.base(u))

    def predict_next(self, u):
        return self(u)

    def learned_state_propagation(self):
        return dict(
            scope="complete prediction, including its increment",
            dc_amplitude=float(self.dc_amplitude.detach()),
            dc_gain=float(torch.exp(-self.dc_amplitude.detach().square())),
            axis_amplitude=self.axis_amplitude.detach().cpu().tolist(),
            axis_quadratic_coefficient=self.axis_amplitude.detach()
            .square()
            .cpu()
            .tolist(),
            axis_nyquist_gain=torch.exp(
                -self.dc_amplitude.detach().square()
                - self.axis_amplitude.detach().square()
            )
            .cpu()
            .tolist(),
        )


def build_model(ndim, seed=0):
    if seed != 0 or ndim not in SPATIAL_SIZES:
        raise ValueError("fixed all14 seed0 candidate only")
    return CompleteUpdateSGNO(build_reference(ndim, seed), ndim)


def model_spec(m):
    c = model_config(m.ndim)
    elements = sum((p.numel() for p in m.parameters()))
    real = sum((p.numel() * (2 if p.is_complex() else 1) for p in m.parameters()))
    assert (elements, real) == (c["tensor_entries"], c["real_parameters"])
    return dict(
        method=METHOD,
        version=VERSION,
        ndim=m.ndim,
        config=c,
        parameter_counts=dict(tensor_entries=elements, real_parameters=real),
        inherited_initialization="exact original backbone and RNG; deterministic filter parameters",
        prediction="G applied to the complete original next-state prediction",
    )
