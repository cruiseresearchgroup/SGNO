"""Affine modal propagation and its trainable parameterization."""

import hashlib
import numpy as np
import torch
from torch import nn
from sgno._consistency import build_model as build_residual
from sgno._consistency import model_config as residual_config
from sgno._spec import SPATIAL_SIZES, ARCHITECTURES
from sgno._fit import fit, low_mask

METHOD = "sgno_linear_residual_v1"
VERSION = "training-initialized-affine-exponential-propagation-sgno-residual-v1"


def model_config(ndim):
    c = residual_config(ndim)
    n = SPATIAL_SIZES[ndim]
    m = ARCHITECTURES[ndim][1]
    symbol = n ** (ndim - 1) * (n // 2 + 1)
    bias = (2 * m - 1) ** (ndim - 1) * m
    c["tensor_entries"] += symbol + bias + 1
    c["real_parameters"] += 2 * (symbol + bias) + 1
    c["physical_propagation"] = dict(
        version=VERSION,
        formula="L(u) + alpha*(F_hf(u)-u)",
        multiplier="exp(-real(z)^2 + i*imag(z))",
        forcing="complex coefficients in existing retained modes",
        initialization="Constrained affine Fourier least squares on training pairs only; alpha=0",
        joint_training=True,
        physical_coefficients_supplied=False,
        task_switches=False,
        multiplier_elements=symbol,
        forcing_elements=bias,
        residual_gate_elements=1,
    )
    return c


class LinearResidualSGNO(nn.Module):
    def __init__(self, residual, ndim):
        super().__init__()
        self.residual = residual
        self.ndim = ndim
        n = SPATIAL_SIZES[ndim]
        self.spatial_shape = (n,) * ndim
        self.spectral_shape = (n,) * (ndim - 1) + (n // 2 + 1,)
        mask = low_mask(self.spatial_shape, ARCHITECTURES[ndim][1])
        self.register_buffer(
            "bias_indices",
            torch.from_numpy(np.flatnonzero(mask.ravel())),
            persistent=False,
        )
        self.generator = nn.Parameter(
            torch.zeros(self.spectral_shape, dtype=torch.complex64)
        )
        self.forcing = nn.Parameter(torch.zeros(int(mask.sum()), dtype=torch.complex64))
        self.residual_weight = nn.Parameter(torch.zeros(()))

    def multiplier(self):
        return torch.exp(
            torch.complex(-self.generator.real.square(), self.generator.imag)
        )

    def linear(self, u):
        axes = tuple(range(-self.ndim, 0))
        h = torch.fft.rfftn(u.float(), dim=axes, norm="ortho")
        b = torch.zeros(
            self.spectral_shape, device=u.device, dtype=torch.complex64
        ).flatten()
        b = b.scatter(0, self.bias_indices, self.forcing).reshape(self.spectral_shape)
        return torch.fft.irfftn(
            self.multiplier() * h + b, s=self.spatial_shape, dim=axes, norm="ortho"
        ).to(u.dtype)

    @torch.no_grad()
    def initialize_from_train(self, data):
        a, b, receipt = fit(data, ARCHITECTURES[self.ndim][1])
        radius = np.clip(np.abs(a), 1e-12, 1.0)
        z = (np.sqrt(-np.log(radius)) + 1j * np.angle(a)).astype(np.complex64)
        selected = b.ravel()[self.bias_indices.cpu().numpy()].astype(np.complex64)
        self.generator.copy_(torch.from_numpy(z))
        self.forcing.copy_(torch.from_numpy(selected))
        self.residual_weight.zero_()
        receipt.update(
            generator_initial_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
            forcing_initial_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            residual_gate_initial=0.0,
            subsequent_training="All parameters, same one-step MSE and Adam schedule",
        )
        return receipt

    def forward(self, u):
        return self.linear(u) + self.residual_weight * (self.residual(u) - u)

    def predict_next(self, u):
        return self(u)

    def learned_state_propagation(self):
        r = self.residual.learned_state_propagation()
        r.update(
            scope="Affine state propagation plus gated SGNO residual",
            residual_weight=float(self.residual_weight.detach()),
            state_multiplier_max_gain=float(self.multiplier().detach().abs().max()),
            state_multiplier_min_gain=float(self.multiplier().detach().abs().min()),
            forcing_l2=float(torch.linalg.vector_norm(self.forcing.detach())),
        )
        return r


def build_model(ndim, seed=0):
    if ndim not in SPATIAL_SIZES or seed != 0:
        raise ValueError("all14 seed0 development screen")
    return LinearResidualSGNO(build_residual(ndim, seed), ndim)


def model_spec(model):
    c = model_config(model.ndim)
    elements = sum((p.numel() for p in model.parameters()))
    real = sum((p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
    assert (elements, real) == (c["tensor_entries"], c["real_parameters"])
    return dict(
        method=METHOD,
        version=VERSION,
        ndim=model.ndim,
        config=c,
        parameter_counts=dict(tensor_entries=elements, real_parameters=real),
        inherited_initialization="Exact SGNO backbone and RNG; deterministic training-only affine fit",
        prediction="Affine Fourier propagation plus alpha times the SGNO nonlinear increment",
    )
