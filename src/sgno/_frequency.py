"""Signed-frequency radial profiles for the two-dimensional spectral layer."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from sgno import sgno as legacy

SGNO_SIGNED_RADIUS_2D_VERSION = "sgno-2d-signed-radius-union-v1-20260910"
SGNO_LEGACY_1D_VERSION = "sgno-1d-canonical-unchanged"
RADIUS_NORMALIZATION = "sqrt(kx^2+ky^2+1e-12)/sqrt(modes1^2+(modes2-1)^2)"


def signed_frequency_radii_2d(
    modes1: int,
    modes2: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (positive, negative) radii in exactly the retained FFT order."""
    modes1, modes2 = (int(modes1), int(modes2))
    if min(modes1, modes2) < 1:
        raise ValueError("modes1 and modes2 must be positive")
    kx_pos = torch.arange(modes1, device=device, dtype=dtype)
    kx_neg = torch.arange(-modes1, 0, device=device, dtype=dtype)
    ky = torch.arange(modes2, device=device, dtype=dtype)
    normalizer = math.sqrt(float(modes1**2 + (modes2 - 1) ** 2))

    def radius(kx: torch.Tensor) -> torch.Tensor:
        return (
            torch.sqrt(kx[:, None].square() + ky[None, :].square() + 1e-12) / normalizer
        )

    return (radius(kx_pos), radius(kx_neg))


class SpectralETD2dSignedRadius(legacy.SpectralETD2d):
    """Legacy parameter layout and ETD computation with signed 2D radii."""

    model_version = SGNO_SIGNED_RADIUS_2D_VERSION
    radius_normalization = RADIUS_NORMALIZATION

    def _damping_profile(self, device) -> torch.Tensor:
        """Stack both blocks as (2, C, m1, m2), including for gain audits."""
        radii = signed_frequency_radii_2d(self.modes1, self.modes2, device=device)
        return torch.stack(
            [
                legacy._nonnegative_radial_profile(
                    radius.unsqueeze(0),
                    self.damp_bias_raw.to(device=device),
                    self.damp_slope_raw.to(device=device),
                )
                for radius in radii
            ]
        )

    def _injection_budget(self, device) -> torch.Tensor:
        """Stack both blocks as (2, C, m1, m2), including for gain audits."""
        radii = signed_frequency_radii_2d(self.modes1, self.modes2, device=device)
        return torch.stack(
            [
                legacy._unit_centered_radial_budget(
                    radius.unsqueeze(0),
                    self.inj_bias.to(device=device),
                    self.inj_slope_raw.to(device=device),
                )
                for radius in radii
            ]
        )

    def _lambda(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        damp_pos, damp_neg = self._damping_profile(device).unbind(0)
        a_pos = -(F.softplus(self.log_decay_pos.to(device=device)) + damp_pos)
        a_neg = -(F.softplus(self.log_decay_neg.to(device=device)) + damp_neg)
        return (
            torch.complex(a_pos, torch.zeros_like(a_pos)).to(device=device),
            torch.complex(a_neg, torch.zeros_like(a_neg)).to(device=device),
        )

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape != v.shape:
            raise ValueError(
                f"v and g must have the same shape, got {v.shape} and {g.shape}"
            )
        _, c, x_size, y_size = v.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")
        v_hat = torch.fft.rfft2(v.float())
        g_hat = torch.fft.rfft2(g.float())
        out_hat = torch.zeros_like(v_hat)
        if 2 * self.modes1 > x_size:
            raise ValueError(
                f"modes1={self.modes1} makes positive/negative full-axis blocks overlap for N={x_size}"
            )
        if self.modes2 > v_hat.size(-1):
            raise ValueError(
                f"modes2={self.modes2} exceeds {v_hat.size(-1)} stored rFFT bins"
            )
        m1, m2 = (self.modes1, self.modes2)
        lam_pos, lam_neg = self._lambda(v.device)
        z_pos = (legacy._CANONICAL_DT * lam_pos[:, :m1, :m2]).unsqueeze(0)
        z_neg = (legacy._CANONICAL_DT * lam_neg[:, :m1, :m2]).unsqueeze(0)
        budget_pos, budget_neg = self._injection_budget(v.device).unbind(0)
        budget_pos = budget_pos[:, :m1, :m2].view(1, self.channels, m1, m2)
        budget_neg = budget_neg[:, :m1, :m2].view(1, self.channels, m1, m2)
        vp = v_hat[:, :, :m1, :m2]
        gp = g_hat[:, :, :m1, :m2]
        mixp = self._project_mix(self.mix_pos.to(device=v.device))[:, :, :m1, :m2]
        gmp = torch.einsum("bixy,ioxy->boxy", gp, mixp)
        out_hat[:, :, :m1, :m2] = (
            torch.exp(z_pos) * vp
            + legacy._CANONICAL_DT * legacy._phi1(z_pos) * gmp * budget_pos
        )
        vn = v_hat[:, :, -m1:, :m2]
        gn = g_hat[:, :, -m1:, :m2]
        mixn = self._project_mix(self.mix_neg.to(device=v.device))[:, :, :m1, :m2]
        gmn = torch.einsum("bixy,ioxy->boxy", gn, mixn)
        out_hat[:, :, -m1:, :m2] = (
            torch.exp(z_neg) * vn
            + legacy._CANONICAL_DT * legacy._phi1(z_neg) * gmn * budget_neg
        )
        return torch.fft.irfft2(out_hat, s=(x_size, y_size))


class SGNO2dSignedRadius(legacy.SGNO2d):
    """Keep the original model initialization and all non-spectral methods."""

    model_version = SGNO_SIGNED_RADIUS_2D_VERSION
    radius_normalization = RADIUS_NORMALIZATION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.etds:
            block.__class__ = SpectralETD2dSignedRadius


def build_sgno2d_signed_radius(
    *,
    num_channels: int = 1,
    modes1: int = 10,
    modes2: int = 10,
    width: int = 5,
    initial_step: int = 1,
    n_blocks: int = 9,
) -> legacy.SGNOTorchApeWrapper:
    """Construct the unchanged residual wrapper: B,C,X,Y -> B,C,X,Y."""
    core = SGNO2dSignedRadius(
        num_channels=num_channels,
        modes1=modes1,
        modes2=modes2,
        width=width,
        initial_step=initial_step,
        n_blocks=n_blocks,
    )
    model = legacy.SGNOTorchApeWrapper(core, spatial_dim=2, initial_step=initial_step)
    model.model_version = SGNO_SIGNED_RADIUS_2D_VERSION
    model.radius_normalization = RADIUS_NORMALIZATION
    return model


def build_regime_sgno(
    *,
    num_spatial_dims: int,
    num_channels: int = 1,
    width: int | None = None,
    modes: int | None = None,
    n_blocks: int | None = None,
    initial_step: int = 1,
) -> legacy.SGNOTorchApeWrapper:
    """Batch entry point: exact legacy 1D, explicit-version corrected 2D.

    Dimension defaults are the approved starting configurations. 3D is
    deliberately rejected; this batch does not fix or re-evaluate 3D.
    """
    if num_spatial_dims not in (1, 2):
        raise ValueError("this regime batch supports only 1D and corrected 2D")
    defaults = {1: (11, 26, 7), 2: (5, 10, 9)}[num_spatial_dims]
    width = defaults[0] if width is None else int(width)
    modes = defaults[1] if modes is None else int(modes)
    n_blocks = defaults[2] if n_blocks is None else int(n_blocks)
    if num_spatial_dims == 2:
        return build_sgno2d_signed_radius(
            num_channels=num_channels,
            modes1=modes,
            modes2=modes,
            width=width,
            initial_step=initial_step,
            n_blocks=n_blocks,
        )
    model = legacy.build_sgno_from_config(
        f"sgno_canonical;width={width};modes={modes};n_blocks={n_blocks};initial_step={initial_step}",
        num_spatial_dims=1,
        num_points=0,
        num_channels=num_channels,
    )
    model.model_version = SGNO_LEGACY_1D_VERSION
    return model


@torch.no_grad()
def project_sgno_mix_(model: nn.Module) -> None:
    """Optional post-optimizer mixer projection; forward projection is retained.

    This uses the original per-frequency spectral-norm cap and original
    parameter objects. Call only where the training protocol already calls
    for this post-step operation; it is not a new optimizer or regularizer.
    """
    core = model.core if hasattr(model, "core") else model
    if not isinstance(core, (legacy.SGNO1d, legacy.SGNO2d)):
        raise TypeError("expected a 1D/2D SGNO core or its original wrapper")
    for block in core.etds:
        if isinstance(block, legacy.SpectralETD1d):
            block.mix.copy_(legacy._project_mixer(block.mix, block.channels))
        elif isinstance(block, legacy.SpectralETD2d):
            block.mix_pos.copy_(block._project_mix(block.mix_pos))
            block.mix_neg.copy_(block._project_mix(block.mix_neg))
        else:
            raise TypeError("unsupported SGNO spectral block in regime model")
