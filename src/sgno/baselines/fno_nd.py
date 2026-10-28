"""Fourier neural operator components."""

from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import nn

Point = Tuple[int, ...]
Pair = Tuple[Point, Point]
SUPPORTED_ACTIVATIONS = ("relu", "sigmoid", "tanh", "swish", "gelu", "identity")


def _tuple_for_dim(value: int | Sequence[int], ndim: int, name: str) -> Tuple[int, ...]:
    if isinstance(value, int):
        result = (int(value),) * ndim
    else:
        result = tuple((int(v) for v in value))
    if len(result) != ndim:
        raise ValueError(f"{name} must have length {ndim}, got {result}")
    return result


def _conv_class(ndim: int):
    if ndim == 2:
        return nn.Conv2d
    if ndim == 3:
        return nn.Conv3d
    raise ValueError("FNO ND controls support exactly 2D or 3D")


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = str(name).lower()
    if key == "relu":
        return F.relu
    if key == "sigmoid":
        return torch.sigmoid
    if key == "tanh":
        return torch.tanh
    if key == "swish":
        return F.silu
    if key == "gelu":
        return F.gelu
    if key == "identity":
        return lambda x: x
    raise ValueError(
        f"unsupported APEBench activation {name!r}; choose from {SUPPORTED_ACTIVATIONS}"
    )


def parse_apebench_fno_config(network_config: str) -> Dict[str, object]:
    """Parse APEBench's exact ``FNO;MODES;WIDTH;BLOCKS;ACTIVATION`` syntax."""
    parts = [part.strip() for part in str(network_config).split(";")]
    if len(parts) != 5 or parts[0].lower() != "fno":
        raise ValueError("FNO config must be FNO;MODES;WIDTH;BLOCKS;ACTIVATION")
    modes = int(parts[1])
    width = int(parts[2])
    blocks = int(parts[3])
    activation = parts[4].lower()
    _activation(activation)
    if min(modes, width, blocks) < 1:
        raise ValueError("MODES, WIDTH, and BLOCKS must be positive")
    return {
        "modes": modes,
        "width": width,
        "n_blocks": blocks,
        "activation": activation,
    }


def generate_pdequinox_corner_slices(
    ndim: int, modes: int
) -> Tuple[Tuple[slice, ...], ...]:
    """Port PDEquinox v0.1.2 ``generate_modes_slices`` spatial ordering."""
    if ndim not in (2, 3):
        raise ValueError("corner slices are defined here for 2D or 3D")
    m = int(modes)
    if m < 1:
        raise ValueError("modes must be positive")
    full_axis_choices = [(slice(None, m), slice(-m, None)) for _ in range(ndim - 1)]
    corners = []
    choices = [[slice(None, m)]] + list(reversed(full_axis_choices))
    for choice in itertools.product(*choices):
        corners.append(tuple(reversed(choice)))
    return tuple(corners)


def per_channel_pair_functional_spectral_rank(ndim: int, modes: int) -> int:
    """Real rank of one projected PDEquinox spectral channel-pair map."""
    d = int(ndim)
    m = int(modes)
    if d not in (2, 3) or m < 1:
        raise ValueError("rank formula requires D in {2,3} and M >= 1")
    return int(2**d * m**d - (2 * m - 1) ** (d - 1))


def _ravel(point: Point, shape: Sequence[int]) -> int:
    index = 0
    for coordinate, extent in zip(point, shape):
        index = index * int(extent) + int(coordinate)
    return index


@dataclass(frozen=True)
class FNOFunctionalLayout:
    shape: Tuple[int, ...]
    modes: int
    rfft_shape: Tuple[int, ...]
    raw_points: Tuple[Point, ...]
    raw_boundary_points: Tuple[Point, ...]
    interior_points: Tuple[Point, ...]
    boundary_pairs: Tuple[Pair, ...]
    self_points: Tuple[Point, ...]
    pair_raw_counts: Tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_interior(self) -> int:
        return len(self.interior_points)

    @property
    def n_pairs(self) -> int:
        return len(self.boundary_pairs)

    @property
    def n_self(self) -> int:
        return len(self.self_points)

    @property
    def functional_rank_per_channel_pair(self) -> int:
        return 2 * self.n_interior + 2 * self.n_pairs + self.n_self

    @property
    def raw_real_scalars_per_channel_pair(self) -> int:
        return 2**self.ndim * self.modes**self.ndim

    @property
    def structurally_dummy_per_channel_pair(self) -> int:
        return (2 * self.modes - 1) ** (self.ndim - 1)

    def flat_index(self, point: Point) -> int:
        return _ravel(point, self.rfft_shape)


def build_fno_functional_layout(
    shape: Sequence[int], modes: int
) -> FNOFunctionalLayout:
    full_shape = tuple((int(n) for n in shape))
    ndim = len(full_shape)
    m = int(modes)
    if ndim not in (2, 3):
        raise ValueError("formal FNO layout supports exactly 2D or 3D")
    if any((n < 1 for n in full_shape)) or m < 1:
        raise ValueError("shape and modes must be positive")
    for axis, n in enumerate(full_shape):
        if not m < n / 2:
            raise ValueError(
                f"formal FNO layout requires M < N/2 on every axis; axis {axis} has M={m}, N={n}"
            )
    rfft_shape = full_shape[:-1] + (full_shape[-1] // 2 + 1,)
    signed_full = tuple(range(-m, 0)) + tuple(range(0, m))
    full_axes = [tuple((k % n for k in signed_full)) for n in full_shape[:-1]]
    raw_points = tuple(
        sorted(
            (
                tuple(prefix) + (last,)
                for prefix in itertools.product(*full_axes)
                for last in range(m)
            ),
            key=lambda point: _ravel(point, rfft_shape),
        )
    )
    raw_set = set(raw_points)
    expected_raw_points = 2 ** (ndim - 1) * m**ndim
    if len(raw_set) != expected_raw_points:
        raise RuntimeError(
            "PDEquinox corner support overlapped under an allegedly formal configuration"
        )
    interior = tuple((point for point in raw_points if point[-1] != 0))
    raw_boundary = tuple((point for point in raw_points if point[-1] == 0))
    boundary_union = set(raw_boundary)
    for point in raw_boundary:
        partner = tuple((-k % n for k, n in zip(point[:-1], full_shape[:-1]))) + (0,)
        boundary_union.add(partner)
    pairs = []
    self_points = []
    pair_raw_counts = []
    for point in sorted(boundary_union, key=lambda value: _ravel(value, rfft_shape)):
        partner = tuple((-k % n for k, n in zip(point[:-1], full_shape[:-1]))) + (0,)
        if partner == point:
            self_points.append(point)
        elif _ravel(point, rfft_shape) < _ravel(partner, rfft_shape):
            pairs.append((point, partner))
            pair_raw_counts.append(int(point in raw_set) + int(partner in raw_set))
    layout = FNOFunctionalLayout(
        shape=full_shape,
        modes=m,
        rfft_shape=rfft_shape,
        raw_points=raw_points,
        raw_boundary_points=raw_boundary,
        interior_points=interior,
        boundary_pairs=tuple(pairs),
        self_points=tuple(self_points),
        pair_raw_counts=tuple(pair_raw_counts),
    )
    expected_rank = per_channel_pair_functional_spectral_rank(ndim, m)
    if layout.functional_rank_per_channel_pair != expected_rank:
        raise RuntimeError(
            f"functional layout rank {layout.functional_rank_per_channel_pair} != formula {expected_rank}"
        )
    if (
        layout.raw_real_scalars_per_channel_pair
        - layout.functional_rank_per_channel_pair
        != layout.structurally_dummy_per_channel_pair
    ):
        raise RuntimeError(
            "raw-functional rank difference does not match Hermitian nullity"
        )
    if layout.n_self != 1 or layout.self_points != ((0,) * ndim,):
        raise RuntimeError(
            "M < N/2 should have DC as its only self-conjugate functional point"
        )
    if any((count not in (1, 2) for count in layout.pair_raw_counts)):
        raise RuntimeError(
            "every boundary orbit must contain one or two raw-supported points"
        )
    return layout


class ApeBenchSpectralConvND(nn.Module):
    """Raw-parameter PyTorch port of PDEquinox v0.1.2 ``SpectralConv``."""

    def __init__(
        self, in_channels: int, out_channels: int, shape: Sequence[int], modes: int
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.in_channels != self.out_channels:
            raise ValueError(
                "the pinned ClassicFNO uses equal hidden input/output channels; unequal-channel interpretation of its raw tensor is ambiguous"
            )
        self.shape = tuple((int(n) for n in shape))
        self.ndim = len(self.shape)
        self.layout = build_fno_functional_layout(self.shape, modes)
        self.modes = int(modes)
        self.corner_slices = generate_pdequinox_corner_slices(self.ndim, self.modes)
        groups = 2 ** (self.ndim - 1)
        weight_shape = (groups, self.in_channels, self.out_channels) + (
            self.modes,
        ) * self.ndim
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(weight_shape))
        self.weight_imag = nn.Parameter(scale * torch.randn(weight_shape))

    def raw_weight_spectrum(
        self,
        weight_real: Optional[torch.Tensor] = None,
        weight_imag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        real = self.weight_real if weight_real is None else weight_real
        imag = self.weight_imag if weight_imag is None else weight_imag
        if real.shape != self.weight_real.shape or imag.shape != self.weight_imag.shape:
            raise ValueError("raw weights have the wrong shape")
        weight = torch.complex(real, imag)
        spectrum = weight.new_zeros(
            (self.in_channels, self.out_channels, *self.layout.rfft_shape)
        )
        for corner, slices in enumerate(self.corner_slices):
            spectrum[slice(None), slice(None), *slices] = weight[corner]
        return spectrum

    def effective_weight_spectrum(self) -> torch.Tensor:
        raw = self.raw_weight_spectrum()
        dims = tuple(range(-self.ndim, 0))
        kernel = torch.fft.irfftn(raw, s=self.shape, dim=dims)
        return torch.fft.rfftn(kernel, s=self.shape, dim=dims)

    @torch.no_grad()
    def load_raw_weight_spectrum_(
        self, spectrum: torch.Tensor
    ) -> "ApeBenchSpectralConvND":
        expected = (self.in_channels, self.out_channels, *self.layout.rfft_shape)
        if tuple(spectrum.shape) != expected:
            raise ValueError(
                f"expected raw spectrum shape {expected}, got {tuple(spectrum.shape)}"
            )
        for corner, slices in enumerate(self.corner_slices):
            values = spectrum[slice(None), slice(None), *slices]
            self.weight_real[corner].copy_(values.real)
            self.weight_imag[corner].copy_(values.imag)
        return self

    @torch.no_grad()
    def copy_function_from_hermitian_(
        self, source: "HermitianSpectralConvND"
    ) -> "ApeBenchSpectralConvND":
        _validate_compatible_spectral_layers(self, source)
        effective = source.effective_weight_spectrum()
        raw = effective.new_zeros(effective.shape)
        raw_set = set(self.layout.raw_boundary_points)
        for point in self.layout.interior_points:
            raw[slice(None), slice(None), *point] = effective[
                slice(None), slice(None), *point
            ]
        for representative, partner in self.layout.boundary_pairs:
            rep_value = effective[slice(None), slice(None), *representative]
            partner_value = rep_value.conj()
            rep_raw = representative in raw_set
            partner_raw = partner in raw_set
            if rep_raw and partner_raw:
                raw[slice(None), slice(None), *representative] = rep_value
                raw[slice(None), slice(None), *partner] = partner_value
            elif rep_raw:
                raw[slice(None), slice(None), *representative] = 2.0 * rep_value
            elif partner_raw:
                raw[slice(None), slice(None), *partner] = 2.0 * partner_value
            else:
                raise RuntimeError(
                    "functional boundary orbit has no raw-supported member"
                )
        for point in self.layout.self_points:
            raw[slice(None), slice(None), *point] = effective[
                slice(None), slice(None), *point
            ].real
        return self.load_raw_weight_spectrum_(raw)

    def functional_real_dof_audit(self) -> Dict[str, int]:
        pair_count = self.in_channels * self.out_channels
        raw = pair_count * self.layout.raw_real_scalars_per_channel_pair
        dummy = pair_count * self.layout.structurally_dummy_per_channel_pair
        functional = pair_count * self.layout.functional_rank_per_channel_pair
        return {
            "raw_real_scalars": int(raw),
            "structurally_dummy_real_scalars": int(dummy),
            "functional_real_scalar_degrees_of_freedom": int(functional),
            "per_channel_pair_operator_rank": int(
                self.layout.functional_rank_per_channel_pair
            ),
            "functional_spectral_operator_rank": int(functional),
        }

    def effective_initialization_variance_audit(self) -> Dict[str, float]:
        variance = float((1.0 / (self.in_channels * self.out_channels)) ** 2)
        return {
            "raw_real_or_imag_variance": variance,
            "interior_effective_real_or_imag_variance": variance,
            "boundary_both_raw_effective_real_or_imag_variance": variance / 2.0,
            "boundary_edge_effective_real_or_imag_variance": variance / 4.0,
            "dc_effective_real_variance": variance,
            "dc_imaginary_variance": 0.0,
        }

    def raw_output_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        expected = (x.shape[0], self.in_channels, *self.shape)
        if tuple(x.shape) != expected:
            raise ValueError(
                f"expected B,{self.in_channels},{self.shape}; got {tuple(x.shape)}"
            )
        fft_input = x if x.dtype in (torch.float32, torch.float64) else x.float()
        dims = tuple(range(-self.ndim, 0))
        x_hat = torch.fft.rfftn(fft_input, dim=dims)
        weight = torch.complex(self.weight_real, self.weight_imag).to(dtype=x_hat.dtype)
        out = x_hat.new_zeros((x.shape[0], self.out_channels, *self.layout.rfft_shape))
        for corner, slices in enumerate(self.corner_slices):
            x_values = x_hat[slice(None), slice(None), *slices]
            out_values = torch.einsum("bi...,oi...->bo...", x_values, weight[corner])
            out[slice(None), slice(None), *slices] = out_values
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_output_spectrum(x)
        dims = tuple(range(-self.ndim, 0))
        return torch.fft.irfftn(raw, s=self.shape, dim=dims).to(dtype=x.dtype)


class ProjectedRawSpectralConvND(ApeBenchSpectralConvND):
    """Raw overparameterisation with an explicit orthogonal Hermitian projection."""

    def __init__(
        self, in_channels: int, out_channels: int, shape: Sequence[int], modes: int
    ):
        super().__init__(in_channels, out_channels, shape, modes)
        self.register_buffer(
            "projection_interior_targets",
            torch.tensor(
                [
                    self.layout.flat_index(point)
                    for point in self.layout.interior_points
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "projection_pair_rep_targets",
            torch.tensor(
                [
                    self.layout.flat_index(pair[0])
                    for pair in self.layout.boundary_pairs
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "projection_pair_partner_targets",
            torch.tensor(
                [
                    self.layout.flat_index(pair[1])
                    for pair in self.layout.boundary_pairs
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "projection_self_targets",
            torch.tensor(
                [self.layout.flat_index(point) for point in self.layout.self_points],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def project_rfft_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        if tuple(spectrum.shape[-self.ndim :]) != self.layout.rfft_shape:
            raise ValueError("spectrum has the wrong rFFT shape")
        leading = spectrum.shape[: -self.ndim]
        raw = spectrum.reshape(*leading, -1)
        projected = torch.zeros_like(raw)
        interior = raw.index_select(-1, self.projection_interior_targets)
        projected = projected.index_copy(-1, self.projection_interior_targets, interior)
        representative_raw = raw.index_select(-1, self.projection_pair_rep_targets)
        partner_raw = raw.index_select(-1, self.projection_pair_partner_targets)
        representative = 0.5 * (representative_raw + partner_raw.conj())
        projected = projected.index_copy(
            -1, self.projection_pair_rep_targets, representative
        )
        projected = projected.index_copy(
            -1, self.projection_pair_partner_targets, representative.conj()
        )
        self_values = raw.index_select(-1, self.projection_self_targets).real
        projected = projected.index_copy(
            -1, self.projection_self_targets, self_values.to(dtype=raw.dtype)
        )
        return projected.reshape(*leading, *self.layout.rfft_shape)

    def effective_weight_spectrum(self) -> torch.Tensor:
        return self.project_rfft_spectrum(self.raw_weight_spectrum())

    def projected_output_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_rfft_spectrum(self.raw_output_spectrum(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projected_output_spectrum(x)
        dims = tuple(range(-self.ndim, 0))
        return torch.fft.irfftn(projected, s=self.shape, dim=dims).to(dtype=x.dtype)


class HermitianSpectralConvND(nn.Module):
    """No-null parameterisation of the projected PDEquinox spectral class."""

    def __init__(
        self, in_channels: int, out_channels: int, shape: Sequence[int], modes: int
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if self.in_channels != self.out_channels:
            raise ValueError(
                "Hermitian FNO is matched to equal-width ClassicFNO spectral blocks"
            )
        self.shape = tuple((int(n) for n in shape))
        self.ndim = len(self.shape)
        self.layout = build_fno_functional_layout(self.shape, modes)
        self.modes = int(modes)
        scale = 1.0 / (self.in_channels * self.out_channels)
        self.interior_real = nn.Parameter(
            scale
            * torch.randn(self.in_channels, self.out_channels, self.layout.n_interior)
        )
        self.interior_imag = nn.Parameter(
            scale
            * torch.randn(self.in_channels, self.out_channels, self.layout.n_interior)
        )
        pair_stds = torch.tensor(
            [
                scale / math.sqrt(2.0) if count == 2 else scale / 2.0
                for count in self.layout.pair_raw_counts
            ],
            dtype=torch.float32,
        )
        self.register_buffer("pair_initialization_std", pair_stds, persistent=False)
        self.boundary_pair_real = nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, self.layout.n_pairs)
            * pair_stds.view(1, 1, -1)
        )
        self.boundary_pair_imag = nn.Parameter(
            torch.randn(self.in_channels, self.out_channels, self.layout.n_pairs)
            * pair_stds.view(1, 1, -1)
        )
        self.boundary_self_real = nn.Parameter(
            scale * torch.randn(self.in_channels, self.out_channels, self.layout.n_self)
        )
        self._register_index_buffers()

    def _register_index_buffers(self) -> None:
        layout = self.layout
        self.register_buffer(
            "interior_targets",
            torch.tensor(
                [layout.flat_index(point) for point in layout.interior_points],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "pair_rep_targets",
            torch.tensor(
                [layout.flat_index(pair[0]) for pair in layout.boundary_pairs],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "pair_partner_targets",
            torch.tensor(
                [layout.flat_index(pair[1]) for pair in layout.boundary_pairs],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "self_targets",
            torch.tensor(
                [layout.flat_index(point) for point in layout.self_points],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "pair_raw_counts",
            torch.tensor(layout.pair_raw_counts, dtype=torch.long),
            persistent=False,
        )

    def effective_weight_spectrum(
        self,
        interior_real: Optional[torch.Tensor] = None,
        interior_imag: Optional[torch.Tensor] = None,
        pair_real: Optional[torch.Tensor] = None,
        pair_imag: Optional[torch.Tensor] = None,
        self_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ir = self.interior_real if interior_real is None else interior_real
        ii = self.interior_imag if interior_imag is None else interior_imag
        pr = self.boundary_pair_real if pair_real is None else pair_real
        pi = self.boundary_pair_imag if pair_imag is None else pair_imag
        sr = self.boundary_self_real if self_real is None else self_real
        interior = torch.complex(ir, ii)
        pairs = torch.complex(pr, pi)
        leading = (self.in_channels, self.out_channels)
        flat = interior.new_zeros((*leading, math.prod(self.layout.rfft_shape)))
        flat = flat.index_copy(-1, self.interior_targets, interior)
        flat = flat.index_copy(-1, self.pair_rep_targets, pairs)
        flat = flat.index_copy(-1, self.pair_partner_targets, pairs.conj())
        flat = flat.index_copy(-1, self.self_targets, sr.to(dtype=interior.dtype))
        return flat.reshape(*leading, *self.layout.rfft_shape)

    @torch.no_grad()
    def copy_function_from_apebench_(
        self, source: ApeBenchSpectralConvND
    ) -> "HermitianSpectralConvND":
        _validate_compatible_spectral_layers(self, source)
        effective = source.effective_weight_spectrum()
        flat = effective.reshape(self.in_channels, self.out_channels, -1)
        interior = flat.index_select(-1, self.interior_targets)
        pairs = flat.index_select(-1, self.pair_rep_targets)
        self_values = flat.index_select(-1, self.self_targets)
        self.interior_real.copy_(interior.real)
        self.interior_imag.copy_(interior.imag)
        self.boundary_pair_real.copy_(pairs.real)
        self.boundary_pair_imag.copy_(pairs.imag)
        self.boundary_self_real.copy_(self_values.real)
        return self

    def functional_real_dof_audit(self) -> Dict[str, int]:
        pair_count = self.in_channels * self.out_channels
        functional = pair_count * self.layout.functional_rank_per_channel_pair
        return {
            "raw_real_scalars": int(functional),
            "structurally_dummy_real_scalars": 0,
            "functional_real_scalar_degrees_of_freedom": int(functional),
            "per_channel_pair_operator_rank": int(
                self.layout.functional_rank_per_channel_pair
            ),
            "functional_spectral_operator_rank": int(functional),
        }

    def effective_initialization_variance_audit(self) -> Dict[str, float]:
        variance = float((1.0 / (self.in_channels * self.out_channels)) ** 2)
        return {
            "interior_effective_real_or_imag_variance": variance,
            "boundary_both_raw_effective_real_or_imag_variance": variance / 2.0,
            "boundary_edge_effective_real_or_imag_variance": variance / 4.0,
            "dc_effective_real_variance": variance,
            "dc_imaginary_variance": 0.0,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (x.shape[0], self.in_channels, *self.shape)
        if tuple(x.shape) != expected:
            raise ValueError(
                f"expected B,{self.in_channels},{self.shape}; got {tuple(x.shape)}"
            )
        fft_input = x if x.dtype in (torch.float32, torch.float64) else x.float()
        dims = tuple(range(-self.ndim, 0))
        x_hat = torch.fft.rfftn(fft_input, dim=dims)
        weight = self.effective_weight_spectrum().to(dtype=x_hat.dtype)
        batch = x_hat.shape[0]
        out_flat = torch.einsum(
            "bik,oik->bok",
            x_hat.reshape(batch, self.in_channels, -1),
            weight.reshape(self.out_channels, self.in_channels, -1),
        )
        out_hat = out_flat.reshape(batch, self.out_channels, *self.layout.rfft_shape)
        return torch.fft.irfftn(out_hat, s=self.shape, dim=dims).to(dtype=x.dtype)


def _validate_compatible_spectral_layers(left: nn.Module, right: nn.Module) -> None:
    attributes = ("in_channels", "out_channels", "shape", "modes")
    if any((getattr(left, name) != getattr(right, name) for name in attributes)):
        raise ValueError("spectral layers are not function-class compatible")


class _FNOBlockND(nn.Module):
    def __init__(self, spectral: nn.Module, width: int, ndim: int, activation: str):
        super().__init__()
        self.spectral = spectral
        self.bypass = _conv_class(ndim)(width, width, 1, bias=True)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _activation(self.activation_name)(self.spectral(x) + self.bypass(x))


class _BaseFNOND(nn.Module):
    spectral_class = None

    def __init__(
        self,
        *,
        spatial_shape: Sequence[int],
        num_channels: int,
        width: int,
        modes: int,
        n_blocks: int,
        activation: str,
        outer_residual: bool,
    ):
        super().__init__()
        self.spatial_shape = tuple((int(n) for n in spatial_shape))
        self.ndim = len(self.spatial_shape)
        self.num_channels = int(num_channels)
        self.width = int(width)
        self.modes = int(modes)
        self.n_blocks = int(n_blocks)
        self.activation_name = str(activation).lower()
        self.outer_residual = bool(outer_residual)
        if self.ndim not in (2, 3):
            raise ValueError("FNO ND controls support exactly 2D or 3D")
        if min(self.num_channels, self.width, self.modes, self.n_blocks) < 1:
            raise ValueError("channels, width, modes, and blocks must be positive")
        _activation(self.activation_name)
        build_fno_functional_layout(self.spatial_shape, self.modes)
        conv = _conv_class(self.ndim)
        self.lift = conv(self.num_channels, self.width, 1, bias=True)
        assert self.spectral_class is not None
        self.blocks = nn.ModuleList(
            [
                _FNOBlockND(
                    self.spectral_class(
                        self.width, self.width, self.spatial_shape, self.modes
                    ),
                    self.width,
                    self.ndim,
                    self.activation_name,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.project = conv(self.width, self.num_channels, 1, bias=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        expected = (u.shape[0], self.num_channels, *self.spatial_shape)
        if tuple(u.shape) != expected:
            raise ValueError(
                f"expected B,{self.num_channels},{self.spatial_shape}; got {tuple(u.shape)}"
            )
        x = self.lift(u)
        for block in self.blocks:
            x = block(x)
        output = self.project(x)
        return u + output if self.outer_residual else output

    def functional_real_dof_audit(self) -> Dict[str, int]:
        raw_actual = sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
        spectral_audits = [
            block.spectral.functional_real_dof_audit() for block in self.blocks
        ]
        spectral_raw = sum((audit["raw_real_scalars"] for audit in spectral_audits))
        dummy = sum(
            (audit["structurally_dummy_real_scalars"] for audit in spectral_audits)
        )
        spectral_functional = sum(
            (
                audit["functional_real_scalar_degrees_of_freedom"]
                for audit in spectral_audits
            )
        )
        non_spectral = raw_actual - spectral_raw
        functional = non_spectral + spectral_functional
        rank_per_pair = per_channel_pair_functional_spectral_rank(self.ndim, self.modes)
        return {
            "raw_real_scalars": int(raw_actual),
            "structurally_dummy_real_scalars": int(dummy),
            "functional_real_scalar_degrees_of_freedom": int(functional),
            "per_channel_pair_operator_rank": int(rank_per_pair),
            "functional_spectral_operator_rank": int(spectral_functional),
            "non_spectral_real_scalars": int(non_spectral),
        }


class ApeBenchFNOND(_BaseFNOND):
    spectral_class = ApeBenchSpectralConvND

    @torch.no_grad()
    def copy_function_from_hermitian_(
        self, source: "HermitianFNOND"
    ) -> "ApeBenchFNOND":
        _validate_compatible_fno_models(self, source)
        self.lift.load_state_dict(source.lift.state_dict())
        self.project.load_state_dict(source.project.state_dict())
        for target_block, source_block in zip(self.blocks, source.blocks):
            target_block.bypass.load_state_dict(source_block.bypass.state_dict())
            target_block.spectral.copy_function_from_hermitian_(source_block.spectral)
        return self


class ProjectedRawFNOND(_BaseFNOND):
    spectral_class = ProjectedRawSpectralConvND

    @torch.no_grad()
    def copy_parameters_from_apebench_(
        self, source: "ApeBenchFNOND"
    ) -> "ProjectedRawFNOND":
        _validate_compatible_fno_models(self, source)
        self.load_state_dict(source.state_dict())
        return self


class HermitianFNOND(_BaseFNOND):
    spectral_class = HermitianSpectralConvND

    @torch.no_grad()
    def copy_function_from_apebench_(self, source: ApeBenchFNOND) -> "HermitianFNOND":
        _validate_compatible_fno_models(self, source)
        self.lift.load_state_dict(source.lift.state_dict())
        self.project.load_state_dict(source.project.state_dict())
        for target_block, source_block in zip(self.blocks, source.blocks):
            target_block.bypass.load_state_dict(source_block.bypass.state_dict())
            target_block.spectral.copy_function_from_apebench_(source_block.spectral)
        return self


def _validate_compatible_fno_models(left: _BaseFNOND, right: _BaseFNOND) -> None:
    attributes = (
        "spatial_shape",
        "num_channels",
        "width",
        "modes",
        "n_blocks",
        "activation_name",
        "outer_residual",
    )
    if any((getattr(left, name) != getattr(right, name) for name in attributes)):
        raise ValueError("FNO models are not function-class compatible")


def expected_fno_real_dof(
    *,
    ndim: int,
    num_channels: int,
    width: int,
    modes: int,
    n_blocks: int,
    parameterization: str,
) -> Dict[str, int]:
    """Analytic raw, dummy, and functional real-DOF formulas for an FNO."""
    d = int(ndim)
    c = int(num_channels)
    w = int(width)
    m = int(modes)
    b = int(n_blocks)
    if d not in (2, 3) or min(c, w, m, b) < 1:
        raise ValueError("invalid FNO dimensions")
    rank = per_channel_pair_functional_spectral_rank(d, m)
    standard_spectral_raw = w * w * 2**d * m**d
    spectral_functional = w * w * rank
    dummy_per_block = standard_spectral_raw - spectral_functional
    non_spectral = c * w + w + b * (w * w + w) + (w * c + c)
    if parameterization in {"apebench_raw", "projected_raw"}:
        raw = non_spectral + b * standard_spectral_raw
        dummy = b * dummy_per_block
    elif parameterization == "hermitian_functional":
        raw = non_spectral + b * spectral_functional
        dummy = 0
    else:
        raise ValueError(
            "parameterization must be 'apebench_raw', 'projected_raw', or 'hermitian_functional'"
        )
    functional = non_spectral + b * spectral_functional
    return {
        "raw_real_scalars": int(raw),
        "structurally_dummy_real_scalars": int(dummy),
        "functional_real_scalar_degrees_of_freedom": int(functional),
        "per_channel_pair_operator_rank": int(rank),
        "functional_spectral_operator_rank": int(b * spectral_functional),
        "non_spectral_real_scalars": int(non_spectral),
    }


FNO_CAPACITY_CANDIDATES = {
    "main_2d": {
        "ndim": 2,
        "modes": 11,
        "width": 7,
        "n_blocks": 4,
        "functional_real_scalar_degrees_of_freedom": 90994,
        "sgno_reference_functional_real_scalar_degrees_of_freedom": 91762,
    },
    "main_3d": {
        "ndim": 3,
        "modes": 6,
        "width": 5,
        "n_blocks": 7,
        "functional_real_scalar_degrees_of_freedom": 281451,
        "sgno_reference_functional_real_scalar_degrees_of_freedom": 282237,
    },
    "paper_reverse_2d": {
        "ndim": 2,
        "modes": 10,
        "width": 6,
        "n_blocks": 4,
        "functional_real_scalar_degrees_of_freedom": 55051,
        "paper_parser_raw_real_scalars": 57787,
    },
    "paper_reverse_3d": {
        "ndim": 3,
        "modes": 5,
        "width": 7,
        "n_blocks": 4,
        "functional_real_scalar_degrees_of_freedom": 180370,
        "paper_parser_raw_real_scalars": 196246,
    },
}


def _resolve_builder_config(
    *,
    network_config: Optional[str],
    width: Optional[int],
    modes: Optional[int],
    n_blocks: Optional[int],
    activation: Optional[str],
) -> Dict[str, object]:
    if network_config is not None:
        if any((value is not None for value in (width, modes, n_blocks, activation))):
            raise ValueError(
                "network_config cannot be combined with explicit FNO hyperparameters"
            )
        return parse_apebench_fno_config(network_config)
    if width is None or modes is None or n_blocks is None:
        raise ValueError(
            "width, modes, and n_blocks are required without network_config"
        )
    activation_name = "gelu" if activation is None else str(activation).lower()
    _activation(activation_name)
    return {
        "width": int(width),
        "modes": int(modes),
        "n_blocks": int(n_blocks),
        "activation": activation_name,
    }


def build_apebench_fno_nd(
    *,
    num_spatial_dims: int,
    num_points: int | Sequence[int],
    num_channels: int,
    width: Optional[int] = None,
    modes: Optional[int] = None,
    n_blocks: Optional[int] = None,
    activation: Optional[str] = None,
    network_config: Optional[str] = None,
    outer_residual: bool = True,
) -> ApeBenchFNOND:
    ndim = int(num_spatial_dims)
    shape = _tuple_for_dim(num_points, ndim, "num_points")
    config = _resolve_builder_config(
        network_config=network_config,
        width=width,
        modes=modes,
        n_blocks=n_blocks,
        activation=activation,
    )
    return ApeBenchFNOND(
        spatial_shape=shape,
        num_channels=num_channels,
        width=int(config["width"]),
        modes=int(config["modes"]),
        n_blocks=int(config["n_blocks"]),
        activation=str(config["activation"]),
        outer_residual=outer_residual,
    )


def build_hermitian_fno_nd(
    *,
    num_spatial_dims: int,
    num_points: int | Sequence[int],
    num_channels: int,
    width: Optional[int] = None,
    modes: Optional[int] = None,
    n_blocks: Optional[int] = None,
    activation: Optional[str] = None,
    network_config: Optional[str] = None,
    outer_residual: bool = True,
) -> HermitianFNOND:
    ndim = int(num_spatial_dims)
    shape = _tuple_for_dim(num_points, ndim, "num_points")
    config = _resolve_builder_config(
        network_config=network_config,
        width=width,
        modes=modes,
        n_blocks=n_blocks,
        activation=activation,
    )
    return HermitianFNOND(
        spatial_shape=shape,
        num_channels=num_channels,
        width=int(config["width"]),
        modes=int(config["modes"]),
        n_blocks=int(config["n_blocks"]),
        activation=str(config["activation"]),
        outer_residual=outer_residual,
    )


def build_projected_raw_fno_nd(
    *,
    num_spatial_dims: int,
    num_points: int | Sequence[int],
    num_channels: int,
    width: Optional[int] = None,
    modes: Optional[int] = None,
    n_blocks: Optional[int] = None,
    activation: Optional[str] = None,
    network_config: Optional[str] = None,
    outer_residual: bool = True,
) -> ProjectedRawFNOND:
    ndim = int(num_spatial_dims)
    shape = _tuple_for_dim(num_points, ndim, "num_points")
    config = _resolve_builder_config(
        network_config=network_config,
        width=width,
        modes=modes,
        n_blocks=n_blocks,
        activation=activation,
    )
    return ProjectedRawFNOND(
        spatial_shape=shape,
        num_channels=num_channels,
        width=int(config["width"]),
        modes=int(config["modes"]),
        n_blocks=int(config["n_blocks"]),
        activation=str(config["activation"]),
        outer_residual=outer_residual,
    )


__all__ = [
    "ApeBenchFNOND",
    "ApeBenchSpectralConvND",
    "FNO_CAPACITY_CANDIDATES",
    "HermitianFNOND",
    "HermitianSpectralConvND",
    "ProjectedRawFNOND",
    "ProjectedRawSpectralConvND",
    "build_apebench_fno_nd",
    "build_fno_functional_layout",
    "build_hermitian_fno_nd",
    "build_projected_raw_fno_nd",
    "expected_fno_real_dof",
    "generate_pdequinox_corner_slices",
    "parse_apebench_fno_config",
    "per_channel_pair_functional_spectral_rank",
]
