from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


_CANONICAL_DT = 1.0
_MIXER_NORM_CAP = 1.0
_HEAD_WIDTH = 128


def _inv_softplus(x: float, eps: float = 1e-6) -> float:
    x = float(max(x, eps))
    return math.log(math.expm1(x))


def _phi1(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Stable complex-safe evaluation of phi_1 with a finite gradient at zero."""

    small = torch.abs(z) < eps
    z_safe = torch.where(small, torch.ones_like(z), z)
    ratio = torch.expm1(z) / z_safe
    series = (
        1.0
        + z / 2.0
        + z.square() / 6.0
        + z.pow(3) / 24.0
        + z.pow(4) / 120.0
        + z.pow(5) / 720.0
    )
    return torch.where(small, series, ratio)


def _project_rfft_hermitian_boundaries(
    spectrum: torch.Tensor,
    spatial_shape: tuple[int, ...],
) -> torch.Tensor:
    """Project the stored DC/Nyquist planes onto the real-FFT range.

    A multidimensional one-sided real FFT only has an in-storage conjugacy
    constraint on the last-axis DC plane and, for even grids, its Nyquist
    plane.  ``irfftn`` applies this projection implicitly.  Making it explicit
    is provided as an audit/reference operation: the production branch relies
    on the identical implicit projection in ``irfft`` to avoid a redundant
    boundary copy.  It preserves the legacy parameter layout, checkpoints,
    effective forward map, and parameter count.
    """

    shape = tuple(int(size) for size in spatial_shape)
    if not shape or any(size < 1 for size in shape):
        raise ValueError("spatial_shape must contain positive axis lengths")
    expected_spectral_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    if tuple(spectrum.shape[-len(shape) :]) != expected_spectral_shape:
        raise ValueError(
            "stored rFFT shape does not match the requested spatial shape: "
            f"got {tuple(spectrum.shape[-len(shape):])}, expected {expected_spectral_shape}"
        )

    boundaries = [0]
    if shape[-1] % 2 == 0:
        boundaries.append(shape[-1] // 2)

    projected_planes = []
    num_full_axes = len(shape) - 1
    for boundary in boundaries:
        plane = spectrum.select(-1, boundary)
        partner = plane
        first_full_axis = plane.ndim - num_full_axes
        for offset, axis_size in enumerate(shape[:-1]):
            indices = torch.remainder(
                -torch.arange(axis_size, device=spectrum.device),
                axis_size,
            )
            partner = partner.index_select(first_full_axis + offset, indices)
        projected_planes.append(0.5 * (plane + partner.conj()))

    boundary_indices = torch.tensor(boundaries, device=spectrum.device, dtype=torch.long)
    source = torch.stack(projected_planes, dim=-1)
    return torch.index_copy(spectrum, -1, boundary_indices, source)


def _radial_coord_nd(shape, device, dtype=torch.float32):
    if not shape:
        raise ValueError("shape must be non-empty")
    grids = []
    for m in shape:
        if m <= 1:
            grids.append(torch.zeros(m, device=device, dtype=dtype))
        else:
            grids.append(torch.linspace(0.0, 1.0, steps=m, device=device, dtype=dtype))
    mesh = torch.meshgrid(*grids, indexing="ij")
    r2 = torch.zeros_like(mesh[0])
    for g in mesh:
        r2 = r2 + g * g
    return torch.sqrt(r2 + 1e-12) / math.sqrt(float(len(shape)))


def _nonnegative_radial_profile(radius: torch.Tensor, bias_raw: torch.Tensor, slope_raw: torch.Tensor):
    slope = F.softplus(slope_raw)
    return F.softplus(bias_raw - slope * radius)


def _unit_centered_radial_budget(radius: torch.Tensor, bias: torch.Tensor, slope_raw: torch.Tensor):
    slope = F.softplus(slope_raw)
    return 2.0 * torch.sigmoid(bias - slope * radius)


def _project_mixer(mix: torch.Tensor, channels: int) -> torch.Tensor:
    mats = mix.reshape(channels, channels, -1).permute(2, 0, 1)
    norms = torch.linalg.matrix_norm(mats, ord=2)
    scales = torch.clamp(_MIXER_NORM_CAP / torch.clamp(norms, min=1e-12), max=1.0)
    mats = mats * scales[:, None, None]
    return mats.permute(1, 2, 0).reshape_as(mix)


class PointwiseMLP1d(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        conv1 = nn.Conv1d(width, 2 * width, 1)
        conv2 = nn.Conv1d(2 * width, width, 1)
        conv1 = nn.utils.parametrizations.spectral_norm(conv1)
        conv2 = nn.utils.parametrizations.spectral_norm(conv2)
        self.net = nn.Sequential(
            conv1,
            nn.GELU(),
            conv2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointwiseMLP2d(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        conv1 = nn.Conv2d(width, 2 * width, 1)
        conv2 = nn.Conv2d(2 * width, width, 1)
        conv1 = nn.utils.parametrizations.spectral_norm(conv1)
        conv2 = nn.utils.parametrizations.spectral_norm(conv2)
        self.net = nn.Sequential(
            conv1,
            nn.GELU(),
            conv2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointwiseMLP3d(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        conv1 = nn.Conv3d(width, 2 * width, 1)
        conv2 = nn.Conv3d(2 * width, width, 1)
        conv1 = nn.utils.parametrizations.spectral_norm(conv1)
        conv2 = nn.utils.parametrizations.spectral_norm(conv2)
        self.net = nn.Sequential(
            conv1,
            nn.GELU(),
            conv2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralETD1d(nn.Module):
    def __init__(self, channels: int, modes1: int):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        if self.channels < 1 or self.modes1 < 1:
            raise ValueError("channels and modes1 must be positive")
        self.dt = _CANONICAL_DT

        self.log_decay = nn.Parameter(torch.randn(channels, modes1) * 0.1)
        scale = 1.0 / (channels * channels)
        self.mix = nn.Parameter(scale * torch.rand(channels, channels, modes1, dtype=torch.cfloat))

        tiny = _inv_softplus(1e-3)
        self.damp_bias_raw = nn.Parameter(torch.full((channels, 1), tiny, dtype=torch.float32))
        self.damp_slope_raw = nn.Parameter(torch.full((channels, 1), tiny, dtype=torch.float32))
        self.inj_bias = nn.Parameter(torch.zeros(channels, 1, dtype=torch.float32))
        self.inj_slope_raw = nn.Parameter(torch.full((channels, 1), tiny, dtype=torch.float32))

    def _damping_profile(self, device):
        radius = _radial_coord_nd((self.modes1,), device=device).view(1, self.modes1)
        return _nonnegative_radial_profile(
            radius,
            self.damp_bias_raw.to(device=device),
            self.damp_slope_raw.to(device=device),
        )

    def _injection_budget(self, device):
        radius = _radial_coord_nd((self.modes1,), device=device).view(1, self.modes1)
        return _unit_centered_radial_budget(
            radius,
            self.inj_bias.to(device=device),
            self.inj_slope_raw.to(device=device),
        )

    def _lambda(self, device):
        log_decay = self.log_decay.to(device=device)
        alpha = -(F.softplus(log_decay) + self._damping_profile(device))
        beta = torch.zeros_like(alpha)
        return torch.complex(alpha, beta).to(device=device)

    def _mix_projected(self, device):
        return _project_mixer(self.mix.to(device=device), self.channels)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape != v.shape:
            raise ValueError(f"v and g must have the same shape, got {v.shape} and {g.shape}")
        _, c, x_size = v.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")

        v_hat = torch.fft.rfft(v.float())
        g_hat = torch.fft.rfft(g.float())
        out_hat = torch.zeros_like(v_hat)

        if self.modes1 > v_hat.size(-1):
            raise ValueError(
                f"modes1={self.modes1} exceeds {v_hat.size(-1)} stored rFFT bins"
            )
        m1 = self.modes1
        lam = self._lambda(v.device)[:, :m1]
        z = (_CANONICAL_DT * lam).unsqueeze(0)

        vh = v_hat[:, :, :m1]
        gh = g_hat[:, :, :m1]
        mix = self._mix_projected(v.device)[:, :, :m1]
        gmix = torch.einsum("bim,iom->bom", gh, mix)

        lin = torch.exp(z) * vh
        forcing = (_CANONICAL_DT * _phi1(z)) * gmix
        budget = self._injection_budget(v.device)[:, :m1].view(1, self.channels, m1)
        out_hat[:, :, :m1] = lin + forcing * budget
        # irfft realizes F_r^{-1} P_H: imaginary self-conjugate components are
        # projected away without a redundant explicit boundary copy.
        return torch.fft.irfft(out_hat, n=x_size)


class SpectralETD2d(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if self.channels < 1 or min(self.modes1, self.modes2) < 1:
            raise ValueError("channels, modes1, and modes2 must be positive")
        self.dt = _CANONICAL_DT

        self.log_decay_pos = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)
        self.log_decay_neg = nn.Parameter(torch.randn(channels, modes1, modes2) * 0.1)

        scale = 1.0 / math.sqrt(channels)
        self.mix_pos = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2, dtype=torch.cfloat))
        self.mix_neg = nn.Parameter(scale * torch.randn(channels, channels, modes1, modes2, dtype=torch.cfloat))

        tiny = _inv_softplus(1e-3)
        self.damp_bias_raw = nn.Parameter(torch.full((channels, 1, 1), tiny, dtype=torch.float32))
        self.damp_slope_raw = nn.Parameter(torch.full((channels, 1, 1), tiny, dtype=torch.float32))
        self.inj_bias = nn.Parameter(torch.zeros(channels, 1, 1, dtype=torch.float32))
        self.inj_slope_raw = nn.Parameter(torch.full((channels, 1, 1), tiny, dtype=torch.float32))

    def _damping_profile(self, device):
        radius = _radial_coord_nd((self.modes1, self.modes2), device=device).view(1, self.modes1, self.modes2)
        return _nonnegative_radial_profile(
            radius,
            self.damp_bias_raw.to(device=device),
            self.damp_slope_raw.to(device=device),
        )

    def _injection_budget(self, device):
        radius = _radial_coord_nd((self.modes1, self.modes2), device=device).view(1, self.modes1, self.modes2)
        return _unit_centered_radial_budget(
            radius,
            self.inj_bias.to(device=device),
            self.inj_slope_raw.to(device=device),
        )

    def _lambda(self, device):
        damp = self._damping_profile(device)
        a_pos = -(F.softplus(self.log_decay_pos.to(device=device)) + damp)
        a_neg = -(F.softplus(self.log_decay_neg.to(device=device)) + damp)
        lam_pos = torch.complex(a_pos, torch.zeros_like(a_pos)).to(device=device)
        lam_neg = torch.complex(a_neg, torch.zeros_like(a_neg)).to(device=device)
        return lam_pos, lam_neg

    def _project_mix(self, mix: torch.Tensor):
        return _project_mixer(mix, self.channels)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape != v.shape:
            raise ValueError(f"v and g must have the same shape, got {v.shape} and {g.shape}")
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
        m1 = self.modes1
        m2 = self.modes2

        lam_pos, lam_neg = self._lambda(v.device)
        lam_pos = lam_pos[:, :m1, :m2]
        lam_neg = lam_neg[:, :m1, :m2]
        z_pos = (_CANONICAL_DT * lam_pos).unsqueeze(0)
        z_neg = (_CANONICAL_DT * lam_neg).unsqueeze(0)

        budget = self._injection_budget(v.device)[:, :m1, :m2].view(1, self.channels, m1, m2)

        vp = v_hat[:, :, :m1, :m2]
        gp = g_hat[:, :, :m1, :m2]
        mixp = self._project_mix(self.mix_pos.to(device=v.device))[:, :, :m1, :m2]
        gmp = torch.einsum("bixy,ioxy->boxy", gp, mixp)
        out_hat[:, :, :m1, :m2] = torch.exp(z_pos) * vp + (_CANONICAL_DT * _phi1(z_pos)) * gmp * budget

        vn = v_hat[:, :, -m1:, :m2]
        gn = g_hat[:, :, -m1:, :m2]
        mixn = self._project_mix(self.mix_neg.to(device=v.device))[:, :, :m1, :m2]
        gmn = torch.einsum("bixy,ioxy->boxy", gn, mixn)
        out_hat[:, :, -m1:, :m2] = torch.exp(z_neg) * vn + (_CANONICAL_DT * _phi1(z_neg)) * gmn * budget

        # irfft2 implicitly applies the Hermitian boundary projection P_H.
        return torch.fft.irfft2(out_hat, s=(x_size, y_size))


class SpectralETD3d(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.channels = int(channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.modes3 = int(modes3)
        if self.channels < 1 or min(self.modes1, self.modes2, self.modes3) < 1:
            raise ValueError("channels and all mode counts must be positive")
        self.dt = _CANONICAL_DT

        shape = (channels, modes1, modes2, modes3)
        self.log_decay1 = nn.Parameter(torch.randn(*shape) * 0.1)
        self.log_decay2 = nn.Parameter(torch.randn(*shape) * 0.1)
        self.log_decay3 = nn.Parameter(torch.randn(*shape) * 0.1)
        self.log_decay4 = nn.Parameter(torch.randn(*shape) * 0.1)

        scale = 1.0 / (channels * channels)
        self.mix1 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix2 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix3 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.mix4 = nn.Parameter(scale * torch.rand(channels, channels, modes1, modes2, modes3, dtype=torch.cfloat))

        tiny = _inv_softplus(1e-3)
        self.damp_bias_raw = nn.Parameter(torch.full((channels, 1, 1, 1), tiny, dtype=torch.float32))
        self.damp_slope_raw = nn.Parameter(torch.full((channels, 1, 1, 1), tiny, dtype=torch.float32))
        self.inj_bias = nn.Parameter(torch.zeros(channels, 1, 1, 1, dtype=torch.float32))
        self.inj_slope_raw = nn.Parameter(torch.full((channels, 1, 1, 1), tiny, dtype=torch.float32))

    def _damping_profile(self, device):
        radius = _radial_coord_nd((self.modes1, self.modes2, self.modes3), device=device).view(
            1,
            self.modes1,
            self.modes2,
            self.modes3,
        )
        return _nonnegative_radial_profile(
            radius,
            self.damp_bias_raw.to(device=device),
            self.damp_slope_raw.to(device=device),
        )

    def _injection_budget(self, device):
        radius = _radial_coord_nd((self.modes1, self.modes2, self.modes3), device=device).view(
            1,
            self.modes1,
            self.modes2,
            self.modes3,
        )
        return _unit_centered_radial_budget(
            radius,
            self.inj_bias.to(device=device),
            self.inj_slope_raw.to(device=device),
        )

    def _lam_block(self, log_decay: torch.Tensor, device):
        damp = self._damping_profile(device)
        alpha = -(F.softplus(log_decay.to(device=device)) + damp)
        return torch.complex(alpha, torch.zeros_like(alpha)).to(device=device)

    def _project_mix(self, mix: torch.Tensor):
        return _project_mixer(mix, self.channels)

    def forward(self, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape != v.shape:
            raise ValueError(f"v and g must have the same shape, got {v.shape} and {g.shape}")
        _, c, x_size, y_size, z_size = v.shape
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")

        v_hat = torch.fft.rfftn(v.float(), dim=[-3, -2, -1])
        g_hat = torch.fft.rfftn(g.float(), dim=[-3, -2, -1])
        out_hat = torch.zeros_like(v_hat)

        if 2 * self.modes1 > x_size or 2 * self.modes2 > y_size:
            raise ValueError(
                "positive/negative full-axis spectral blocks overlap; "
                f"modes=({self.modes1},{self.modes2}) shape=({x_size},{y_size})"
            )
        if self.modes3 > v_hat.size(-1):
            raise ValueError(
                f"modes3={self.modes3} exceeds {v_hat.size(-1)} stored rFFT bins"
            )
        m1 = self.modes1
        m2 = self.modes2
        m3 = self.modes3
        budget = self._injection_budget(v.device)[:, :m1, :m2, :m3].view(1, self.channels, m1, m2, m3)

        def apply_block(vs, gs, lam, mix):
            zc = (_CANONICAL_DT * lam).unsqueeze(0)
            mix = self._project_mix(mix.to(device=v.device))[:, :, :m1, :m2, :m3]
            gmix = torch.einsum("bixyz,ioxyz->boxyz", gs, mix)
            return torch.exp(zc) * vs + (_CANONICAL_DT * _phi1(zc)) * gmix * budget

        lam1 = self._lam_block(self.log_decay1[:, :m1, :m2, :m3], v.device)
        lam2 = self._lam_block(self.log_decay2[:, :m1, :m2, :m3], v.device)
        lam3 = self._lam_block(self.log_decay3[:, :m1, :m2, :m3], v.device)
        lam4 = self._lam_block(self.log_decay4[:, :m1, :m2, :m3], v.device)

        out_hat[:, :, :m1, :m2, :m3] = apply_block(
            v_hat[:, :, :m1, :m2, :m3],
            g_hat[:, :, :m1, :m2, :m3],
            lam1,
            self.mix1,
        )
        out_hat[:, :, -m1:, :m2, :m3] = apply_block(
            v_hat[:, :, -m1:, :m2, :m3],
            g_hat[:, :, -m1:, :m2, :m3],
            lam2,
            self.mix2,
        )
        out_hat[:, :, :m1, -m2:, :m3] = apply_block(
            v_hat[:, :, :m1, -m2:, :m3],
            g_hat[:, :, :m1, -m2:, :m3],
            lam3,
            self.mix3,
        )
        out_hat[:, :, -m1:, -m2:, :m3] = apply_block(
            v_hat[:, :, -m1:, -m2:, :m3],
            g_hat[:, :, -m1:, -m2:, :m3],
            lam4,
            self.mix4,
        )

        # irfftn implicitly applies the Hermitian boundary projection P_H.
        return torch.fft.irfftn(out_hat, s=(x_size, y_size, z_size))


class SGNO1d(nn.Module):
    def __init__(self, num_channels: int, modes: int = 16, width: int = 64, initial_step: int = 1, n_blocks: int = 4):
        super().__init__()
        self.modes1 = int(modes)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.initial_step = int(initial_step)
        self.padding = 0

        self.fc0 = nn.Linear(self.initial_step * num_channels + 1, self.width)
        self.gs = nn.ModuleList([PointwiseMLP1d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList([SpectralETD1d(self.width, self.modes1) for _ in range(self.n_blocks)])
        self.bs = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_blocks)])
        self.fc1 = nn.Linear(self.width, _HEAD_WIDTH)
        self.fc2 = nn.Linear(_HEAD_WIDTH, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 2, 1)
        for i in range(self.n_blocks):
            upd = self.etds[i](x, self.gs[i](x)) + self.bs[i](x)
            x = upd if i == self.n_blocks - 1 else F.gelu(upd)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).unsqueeze(-2)


class SGNO2d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 20,
        initial_step: int = 1,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.initial_step = int(initial_step)
        self.padding = 0

        self.fc0 = nn.Linear(self.initial_step * num_channels + 2, self.width)
        self.gs = nn.ModuleList([PointwiseMLP2d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList([SpectralETD2d(self.width, self.modes1, self.modes2) for _ in range(self.n_blocks)])
        self.bs = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_blocks)])
        self.fc1 = nn.Linear(self.width, _HEAD_WIDTH)
        self.fc2 = nn.Linear(_HEAD_WIDTH, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for i in range(self.n_blocks):
            upd = self.etds[i](x, self.gs[i](x)) + self.bs[i](x)
            x = upd if i == self.n_blocks - 1 else F.gelu(upd)
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).unsqueeze(-2)


class SGNO3d(nn.Module):
    def __init__(
        self,
        num_channels: int,
        modes1: int = 8,
        modes2: int = 8,
        modes3: int = 8,
        width: int = 20,
        initial_step: int = 1,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.modes3 = int(modes3)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.initial_step = int(initial_step)
        self.padding = 0

        self.fc0 = nn.Linear(self.initial_step * num_channels + 3, self.width)
        self.gs = nn.ModuleList([PointwiseMLP3d(self.width) for _ in range(self.n_blocks)])
        self.etds = nn.ModuleList(
            [SpectralETD3d(self.width, self.modes1, self.modes2, self.modes3) for _ in range(self.n_blocks)]
        )
        self.bs = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(self.n_blocks)])
        self.fc1 = nn.Linear(self.width, _HEAD_WIDTH)
        self.fc2 = nn.Linear(_HEAD_WIDTH, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x).permute(0, 4, 1, 2, 3)
        for i in range(self.n_blocks):
            upd = self.etds[i](x, self.gs[i](x)) + self.bs[i](x)
            x = upd if i == self.n_blocks - 1 else F.gelu(upd)
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).unsqueeze(-2)


def _make_grid_1d(batch: int, x_size: int, device, dtype):
    x = torch.linspace(0.0, 1.0, steps=x_size, device=device, dtype=dtype).view(1, x_size, 1)
    return x.expand(batch, x_size, 1)


def _make_grid_2d(batch: int, x_size: int, y_size: int, device, dtype):
    gx = torch.linspace(0.0, 1.0, steps=x_size, device=device, dtype=dtype)
    gy = torch.linspace(0.0, 1.0, steps=y_size, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(1, x_size, y_size, 2)
    return grid.expand(batch, x_size, y_size, 2)


def _make_grid_3d(batch: int, x_size: int, y_size: int, z_size: int, device, dtype):
    gx = torch.linspace(0.0, 1.0, steps=x_size, device=device, dtype=dtype)
    gy = torch.linspace(0.0, 1.0, steps=y_size, device=device, dtype=dtype)
    gz = torch.linspace(0.0, 1.0, steps=z_size, device=device, dtype=dtype)
    xx, yy, zz = torch.meshgrid(gx, gy, gz, indexing="ij")
    grid = torch.stack([xx, yy, zz], dim=-1).view(1, x_size, y_size, z_size, 3)
    return grid.expand(batch, x_size, y_size, z_size, 3)


class SGNOTorchApeWrapper(nn.Module):
    def __init__(self, core: nn.Module, spatial_dim: int, initial_step: int):
        super().__init__()
        self.core = core
        self.spatial_dim = int(spatial_dim)
        self.initial_step = int(initial_step)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        device = u.device
        dtype = u.dtype

        if self.spatial_dim == 1:
            if u.ndim == 3:
                if self.initial_step != 1:
                    raise ValueError("history input is required when initial_step > 1")
                base = u
                x = u.permute(0, 2, 1).contiguous()
            elif u.ndim == 4:
                batch, steps, channels, x_size = u.shape
                if steps != self.initial_step:
                    raise ValueError("history length must equal initial_step")
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 2, 1).contiguous().view(batch, x_size, channels * steps)
            else:
                raise ValueError("1D input must have shape B C X or B T C X")
            grid = _make_grid_1d(x.shape[0], x.shape[1], device, dtype)
            delta = self.core(x, grid).squeeze(-2).permute(0, 2, 1).contiguous()
            return base + delta.to(dtype=base.dtype)

        if self.spatial_dim == 2:
            if u.ndim == 4:
                if self.initial_step != 1:
                    raise ValueError("history input is required when initial_step > 1")
                base = u
                x = u.permute(0, 2, 3, 1).contiguous()
            elif u.ndim == 5:
                batch, steps, channels, x_size, y_size = u.shape
                if steps != self.initial_step:
                    raise ValueError("history length must equal initial_step")
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 4, 2, 1).contiguous().view(batch, x_size, y_size, channels * steps)
            else:
                raise ValueError("2D input must have shape B C X Y or B T C X Y")
            grid = _make_grid_2d(x.shape[0], x.shape[1], x.shape[2], device, dtype)
            delta = self.core(x, grid).squeeze(-2).permute(0, 3, 1, 2).contiguous()
            return base + delta.to(dtype=base.dtype)

        if self.spatial_dim == 3:
            if u.ndim == 5:
                if self.initial_step != 1:
                    raise ValueError("history input is required when initial_step > 1")
                base = u
                x = u.permute(0, 2, 3, 4, 1).contiguous()
            elif u.ndim == 6:
                batch, steps, channels, x_size, y_size, z_size = u.shape
                if steps != self.initial_step:
                    raise ValueError("history length must equal initial_step")
                base = u[:, -1].contiguous()
                x = u.permute(0, 3, 4, 5, 2, 1).contiguous().view(
                    batch,
                    x_size,
                    y_size,
                    z_size,
                    channels * steps,
                )
            else:
                raise ValueError("3D input must have shape B C X Y Z or B T C X Y Z")
            grid = _make_grid_3d(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device, dtype)
            delta = self.core(x, grid).squeeze(-2).permute(0, 4, 1, 2, 3).contiguous()
            return base + delta.to(dtype=base.dtype)

        raise ValueError("num_spatial_dims must be 1, 2, or 3")


def _parse_value(value: str):
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+(\.\d+)?([eE]-?\d+)?", value):
        return float(value)
    return value


def _parse_network_config(network_config: str) -> Dict[str, Any]:
    parts = [p.strip() for p in str(network_config or "").split(";") if p.strip()]
    if not parts:
        raise ValueError("network_config must start with sgno_canonical")
    if parts[0].lower() != "sgno_canonical":
        raise ValueError("only sgno_canonical is supported")

    cfg: Dict[str, Any] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError("sgno_canonical uses key=value parameters only")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        if key not in {"width", "modes", "n_blocks", "initial_step"}:
            raise ValueError(
                "sgno_canonical exposes only width, modes, n_blocks, and initial_step; "
                f"got {key!r}"
            )
        cfg[key] = _parse_value(value.strip())
    return cfg


def build_sgno_from_config(
    network_config: str,
    num_spatial_dims: int,
    num_points: int,
    num_channels: int,
) -> nn.Module:
    del num_points
    cfg = _parse_network_config(network_config)
    width = int(cfg.get("width", 64))
    modes = int(cfg.get("modes", 16))
    n_blocks = int(cfg.get("n_blocks", 4))
    initial_step = int(cfg.get("initial_step", 1))

    sd = int(num_spatial_dims)
    if sd == 1:
        core = SGNO1d(
            num_channels=num_channels,
            modes=modes,
            width=width,
            initial_step=initial_step,
            n_blocks=n_blocks,
        )
    elif sd == 2:
        core = SGNO2d(
            num_channels=num_channels,
            modes1=modes,
            modes2=modes,
            width=width,
            initial_step=initial_step,
            n_blocks=n_blocks,
        )
    elif sd == 3:
        core = SGNO3d(
            num_channels=num_channels,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            width=width,
            initial_step=initial_step,
            n_blocks=n_blocks,
        )
    else:
        raise ValueError("num_spatial_dims must be 1, 2, or 3")

    return SGNOTorchApeWrapper(core=core, spatial_dim=sd, initial_step=initial_step)


def _flatten_tensor(t: torch.Tensor) -> List[float]:
    return t.detach().reshape(-1).cpu().tolist()


def _summary_stats(values: Iterable[float]) -> Dict[str, float]:
    xs = list(float(v) for v in values)
    if not xs:
        return {"min": float("nan"), "median": float("nan"), "p95": float("nan")}
    x = torch.tensor(xs, dtype=torch.float64)
    return {
        "min": float(torch.min(x).item()),
        "median": float(torch.median(x).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
    }


def _conv1x1_spectral_norm(conv: nn.Module) -> float:
    weight = getattr(conv, "weight", None)
    if weight is None and hasattr(conv, "parametrizations"):
        weight = conv.weight
    if weight is None:
        return float("nan")
    mat = weight.detach().reshape(weight.shape[0], -1).cpu()
    return float(torch.linalg.matrix_norm(mat, ord=2).item())


def summarize_gain_audit(model: nn.Module) -> Dict[str, Any]:
    core = model.core if hasattr(model, "core") else model
    etds = getattr(core, "etds", [])
    gs = getattr(core, "gs", [])

    gamma_vals: List[float] = []
    mu_vals: List[float] = []
    forcing_norm_vals: List[float] = []
    q_delta_vals: List[float] = []
    bypass_norm_vals: List[float] = []
    adaptive_damp_vals: List[float] = []
    adaptive_budget_vals: List[float] = []

    for etd in etds:
        if isinstance(etd, SpectralETD1d):
            lam = etd._lambda("cpu")
            gamma_vals.extend(_flatten_tensor(-lam.real))
            mix = etd._mix_projected("cpu").permute(2, 0, 1)
            mu_vals.extend(_flatten_tensor(torch.linalg.matrix_norm(mix, ord=2)))
            adaptive_budget_vals.extend(_flatten_tensor(etd._injection_budget("cpu")))
            adaptive_damp_vals.extend(_flatten_tensor(etd._damping_profile("cpu")))
        elif isinstance(etd, SpectralETD2d):
            lam_pos, lam_neg = etd._lambda("cpu")
            gamma_vals.extend(_flatten_tensor(-lam_pos.real))
            gamma_vals.extend(_flatten_tensor(-lam_neg.real))
            adaptive_budget_vals.extend(_flatten_tensor(etd._injection_budget("cpu")))
            adaptive_damp_vals.extend(_flatten_tensor(etd._damping_profile("cpu")))
            for mix in [etd._project_mix(etd.mix_pos.to("cpu")), etd._project_mix(etd.mix_neg.to("cpu"))]:
                mats = mix.permute(2, 3, 0, 1).reshape(-1, etd.channels, etd.channels)
                mu_vals.extend(_flatten_tensor(torch.linalg.matrix_norm(mats, ord=2)))
        elif isinstance(etd, SpectralETD3d):
            for log_decay in [etd.log_decay1, etd.log_decay2, etd.log_decay3, etd.log_decay4]:
                lam = etd._lam_block(log_decay.detach().cpu(), "cpu")
                gamma_vals.extend(_flatten_tensor(-lam.real))
            adaptive_budget_vals.extend(_flatten_tensor(etd._injection_budget("cpu")))
            adaptive_damp_vals.extend(_flatten_tensor(etd._damping_profile("cpu")))
            for mix in [
                etd._project_mix(etd.mix1.to("cpu")),
                etd._project_mix(etd.mix2.to("cpu")),
                etd._project_mix(etd.mix3.to("cpu")),
                etd._project_mix(etd.mix4.to("cpu")),
            ]:
                mats = mix.permute(2, 3, 4, 0, 1).reshape(-1, etd.channels, etd.channels)
                mu_vals.extend(_flatten_tensor(torch.linalg.matrix_norm(mats, ord=2)))

    for g in gs:
        for layer in g.modules():
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                forcing_norm_vals.append(_conv1x1_spectral_norm(layer))

    for b in getattr(core, "bs", []) or []:
        bypass_norm_vals.append(_conv1x1_spectral_norm(b))

    if gamma_vals and mu_vals and forcing_norm_vals and etds:
        gamma_min = min(gamma_vals)
        mu_max = max(mu_vals)
        l_n = max(forcing_norm_vals)
        for etd in etds:
            dt = float(etd.dt)
            q = math.exp(-gamma_min * dt) + mu_max * l_n * (1.0 - math.exp(-gamma_min * dt)) / gamma_min
            q_delta_vals.append(float(q))

    q_delta_summary = _summary_stats(q_delta_vals)
    return {
        "scope_warning": (
            "q_delta is a branch-only heuristic proxy. It omits bypass, injection budget, "
            "activation, lift, projection, and the outer residual, so it is not the paper's "
            "full-map Lipschitz bound and cannot certify rollout stability."
        ),
        "certifies_full_map": False,
        "gamma_min": _summary_stats(gamma_vals),
        "mu_max": _summary_stats(mu_vals),
        "forcing_norm_proxy": _summary_stats(forcing_norm_vals),
        "adaptive_damping_profile": _summary_stats(adaptive_damp_vals),
        "adaptive_injection_budget": _summary_stats(adaptive_budget_vals),
        "skip_norm": _summary_stats(bypass_norm_vals),
        "q_delta_heuristic": q_delta_summary,
        "q_delta": q_delta_summary,
    }

