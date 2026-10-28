"""Frozen 2D grid transfer using the evaluated 64-grid extension convention."""

import types
import torch
from torch import nn


def resize_axis(x, size, axis):
    """Periodic Fourier resampling, including even-grid Nyquist splitting."""
    old = x.shape[axis]
    if old == size:
        return x
    spectrum = torch.fft.fft(x.contiguous(), dim=axis)
    shape = list(x.shape)
    shape[axis] = size
    output = torch.zeros(shape, dtype=spectrum.dtype, device=x.device)
    count = min(old, size)

    def section(start, stop):
        index = [slice(None)] * x.ndim
        index[axis] = slice(start, stop)
        return tuple(index)

    positive = (count + 1) // 2
    output[section(0, positive)] = spectrum[section(0, positive)]
    negative = (count - 1) // 2
    if negative:
        output[section(size-negative, size)] = spectrum[section(old-negative, old)]
    if count % 2 == 0:
        if size < old:
            output[section(count//2, count//2+1)] = (
                spectrum[section(count//2, count//2+1)]
                + spectrum[section(old-count//2, old-count//2+1)])
        else:
            nyquist = 0.5 * spectrum[section(count//2, count//2+1)]
            output[section(count//2, count//2+1)] = nyquist
            output[section(size-count//2, size-count//2+1)] = nyquist
    return torch.fft.ifft(output, dim=axis).real * (size / old)


def resize(x, size):
    return resize_axis(resize_axis(x, size, -2), size, -1)


def _linear_on_target(self, u):
    ix, iy, valid = self._transfer_ix, self._transfer_iy, self._transfer_valid
    multiplier = self.multiplier()[ix, iy] * valid
    bias = torch.zeros(64*33, device=u.device, dtype=torch.complex64)
    bias = bias.scatter(0, self.bias_indices, self.forcing).reshape(64, 33)
    bias = bias[ix, iy] * valid * (self._transfer_size / 64)
    spectrum = torch.fft.rfft2(u.float(), norm="ortho")
    return torch.fft.irfft2(spectrum*multiplier+bias,
                           s=(self._transfer_size,)*2, norm="ortho").to(u.dtype)


class FrozenGridPredictor(nn.Module):
    """Own a fresh 64-grid SGNO and freeze it for inference at 32, 64 or 128.

    `frequency_direct` reuses integer Fourier modes and rebuilds runtime buffers.
    `project_to64` resamples to the training grid and back at every prediction.
    Load a new model for each adapter; do not train or checkpoint the adapter.
    """

    def __init__(self, model, grid, policy="frequency_direct"):
        super().__init__()
        if model.ndim != 2 or grid not in (32, 64, 128):
            raise ValueError("This evaluated convention covers 2D grids 32, 64 and 128")
        if policy not in ("frequency_direct", "project_to64"):
            raise ValueError("Unknown transfer policy")
        if hasattr(model, "_transfer_size"):
            raise ValueError("Load a fresh native model for each grid adapter")
        self.model = model.requires_grad_(False).eval()
        self.grid, self.policy = grid, policy
        if grid != 64 and policy == "frequency_direct":
            device = next(model.parameters()).device
            kx = torch.fft.fftfreq(grid, d=1/grid, device=device).round().long()
            ky = torch.arange(grid//2+1, device=device)
            gx, gy = torch.meshgrid(kx, ky, indexing="ij")
            model.register_buffer("_transfer_ix", gx.remainder(64), persistent=False)
            model.register_buffer("_transfer_iy", gy.clamp_max(32), persistent=False)
            model.register_buffer("_transfer_valid", (gx.abs()<32) & (gy<32), persistent=False)
            model._transfer_size = grid
            model.linear = types.MethodType(_linear_on_target, model)
            residual, predictor = model.residual, model.residual.predictor
            residual.spatial_shape = (grid, grid)
            residual.high_mask = (gx.abs()>=residual.cutoff) | (gy>=residual.cutoff)
            residual._eval_correction = None
            predictor.spatial_shape = (grid, grid)
            predictor.normalized_frequency_squared = torch.stack(
                [(2*gx.float()/64)**2, (2*gy.float()/64)**2])
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("FrozenGridPredictor is an inference-only adapter")
        return super().train(False)

    @torch.no_grad()
    def forward(self, state):
        if tuple(state.shape[-2:]) != (self.grid, self.grid):
            raise ValueError("Input grid does not match the adapter")
        if self.policy == "project_to64":
            return resize(self.model(resize(state, 64)), self.grid)
        return self.model(state)
