"""Fourier neural operator components."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch
import torch.nn.functional as F
from torch import nn

MIXER_NORM_CAP = 1.0
HEAD_WIDTH = 128
BRANCH_VARIANTS = (
    "full",
    "no_carry",
    "no_correction",
    "no_bypass",
    "carry_only",
    "correction_only",
    "bypass_only",
)
GENERATOR_TYPES = ("real", "complex", "complex_basis3", "signed_complex")
PHASE_ROUTES = ("both", "carry_only", "correction_only")


@dataclass(frozen=True)
class BranchFlags:
    carry: bool
    correction: bool
    bypass: bool


_BRANCH_FLAGS: Dict[str, BranchFlags] = {
    "full": BranchFlags(True, True, True),
    "no_carry": BranchFlags(False, True, True),
    "no_correction": BranchFlags(True, False, True),
    "no_bypass": BranchFlags(True, True, False),
    "carry_only": BranchFlags(True, False, False),
    "correction_only": BranchFlags(False, True, False),
    "bypass_only": BranchFlags(False, False, True),
}


def branch_flags(name: str) -> BranchFlags:
    try:
        return _BRANCH_FLAGS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown branch variant {name!r}; choose from {BRANCH_VARIANTS}"
        ) from exc


def inv_softplus(x: float, eps: float = 1e-12) -> float:
    x = float(max(x, eps))
    return math.log(math.expm1(x))


def stable_phi1(z: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """Compute phi_1(z)=(exp(z)-1)/z with a finite derivative at z=0."""
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


def count_real_dof(model: nn.Module) -> Dict[str, int]:
    tensor_elements = 0
    real_scalars = 0
    trainable_tensors = 0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        tensor_elements += parameter.numel()
        real_scalars += parameter.numel() * (2 if parameter.is_complex() else 1)
    return {
        "trainable_tensors": trainable_tensors,
        "torch_parameter_elements": tensor_elements,
        "real_scalar_degrees_of_freedom": real_scalars,
    }


def functional_real_dof_audit(model: nn.Module, *, num_points: int) -> Dict[str, int]:
    """Count raw and functionally active real scalars for 1D rFFT models.

    A complex multiplier's imaginary component at the rFFT DC bin (and at the
    Nyquist bin when retained) is discarded by ``irfft``.  The pinned
    PDEquinox-style FNO and the public SGNO allocate those values, so raw
    parameter scalars overstate their functional capacity.  The corrected
    model allocates no such self-conjugate imaginary values.
    """
    counts = count_real_dof(model)
    dummy = 0
    n = int(num_points)
    for module in model.modules():
        if isinstance(module, ApeBenchSpectralConv1d):
            n_self = 1 + int(n % 2 == 0 and module.modes > n // 2)
            dummy += module.in_channels * module.out_channels * n_self
        elif module.__class__.__name__ == "SpectralETD1d":
            mix = getattr(module, "mix", None)
            modes = int(getattr(module, "modes1", 0))
            channels = int(getattr(module, "channels", 0))
            if isinstance(mix, torch.Tensor) and mix.is_complex():
                n_self = 1 + int(n % 2 == 0 and modes > n // 2)
                dummy += channels * channels * n_self
    counts["structurally_dummy_real_scalars"] = int(dummy)
    counts["functional_real_scalar_degrees_of_freedom"] = int(
        counts["real_scalar_degrees_of_freedom"] - dummy
    )
    return counts


def _project_mixer(mix: torch.Tensor, channels: int) -> torch.Tensor:
    matrices = mix.reshape(channels, channels, -1).permute(2, 0, 1)
    norms = torch.linalg.matrix_norm(matrices, ord=2)
    scales = torch.clamp(MIXER_NORM_CAP / torch.clamp(norms, min=1e-12), max=1.0)
    matrices = matrices * scales[:, None, None]
    return matrices.permute(1, 2, 0).reshape_as(mix)


def _nonself_mode_indices(num_points: int, modes: int) -> tuple[int, ...]:
    available = num_points // 2 + 1
    if modes > available:
        raise ValueError(
            f"modes={modes} exceeds {available} rFFT bins for N={num_points}"
        )
    nyquist = num_points // 2 if num_points % 2 == 0 else None
    return tuple((k for k in range(modes) if k != 0 and k != nyquist))


class PointwiseCorrection1d(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        conv1 = nn.utils.parametrizations.spectral_norm(nn.Conv1d(width, 2 * width, 1))
        conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv1d(2 * width, width, 1))
        self.net = nn.Sequential(conv1, nn.GELU(), conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorrectedSpectralGenerator1d(nn.Module):
    """Hermitian-valid scalar ETD branch on a one-sided 1D real FFT."""

    def __init__(
        self,
        channels: int,
        modes: int,
        num_points: int,
        *,
        generator_type: str,
        phase_route: str,
        support_carry: bool,
        support_correction: bool,
        decay_init: str,
    ):
        super().__init__()
        if generator_type not in GENERATOR_TYPES:
            raise ValueError(f"unknown generator_type {generator_type!r}")
        if phase_route not in PHASE_ROUTES:
            raise ValueError(f"unknown phase_route {phase_route!r}")
        if decay_init not in {"canonical", "small"}:
            raise ValueError("decay_init must be 'canonical' or 'small'")
        if not support_carry and (not support_correction):
            raise ValueError(
                "spectral generator needs at least one active spectral branch"
            )
        self.channels = int(channels)
        self.modes = int(modes)
        self.num_points = int(num_points)
        self.generator_type = generator_type
        self.phase_route = phase_route
        self.support_carry = bool(support_carry)
        self.support_correction = bool(support_correction)
        self.nonself_indices = _nonself_mode_indices(self.num_points, self.modes)
        self.register_buffer(
            "nonself_index_tensor",
            torch.tensor(self.nonself_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "imaginary_mode_mask",
            torch.tensor(
                [
                    1.0 if mode in self.nonself_indices else 0.0
                    for mode in range(self.modes)
                ]
            ).view(1, self.modes),
            persistent=False,
        )
        if decay_init == "canonical":
            initial_decay = torch.randn(channels, modes) * 0.1
        else:
            initial_decay = torch.full((channels, modes), inv_softplus(0.001))
        self.log_decay = nn.Parameter(initial_decay)
        tiny = inv_softplus(0.001)
        self.damp_bias_raw = nn.Parameter(torch.full((channels, 1), tiny))
        self.damp_slope_raw = nn.Parameter(torch.full((channels, 1), tiny))
        if generator_type in {"complex", "signed_complex"}:
            self.phase_raw = nn.Parameter(
                torch.zeros(channels, len(self.nonself_indices))
            )
        elif generator_type == "complex_basis3":
            self.phase_coeff = nn.Parameter(torch.zeros(channels, 3))
        if generator_type == "signed_complex":
            self.growth_raw = nn.Parameter(torch.zeros(channels, modes))
        if self.support_correction:
            scale = 1.0 / (channels * channels)
            self.mix_real = nn.Parameter(scale * torch.rand(channels, channels, modes))
            self.mix_imag = nn.Parameter(
                scale * torch.rand(channels, channels, len(self.nonself_indices))
            )
            self.inj_bias = nn.Parameter(torch.zeros(channels, 1))
            self.inj_slope_raw = nn.Parameter(torch.full((channels, 1), tiny))

    def _radius(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        denom = max(1, self.modes - 1)
        return torch.arange(self.modes, device=device, dtype=dtype).view(1, -1) / denom

    def _alpha(self) -> torch.Tensor:
        radius = self._radius(self.log_decay.device, self.log_decay.dtype)
        profile = F.softplus(
            self.damp_bias_raw - F.softplus(self.damp_slope_raw) * radius
        )
        if self.generator_type == "signed_complex":
            return torch.tanh(self.growth_raw) - F.softplus(self.log_decay) - profile
        return -(F.softplus(self.log_decay) + profile)

    def _omega(self) -> torch.Tensor:
        omega = torch.zeros_like(self.log_decay)
        if self.generator_type == "real":
            return omega
        if self.generator_type == "complex_basis3":
            radius = self._radius(self.log_decay.device, self.log_decay.dtype)
            basis = torch.cat((radius, radius.square(), radius.pow(3)), dim=0)
            raw = torch.einsum("cp,pm->cm", self.phase_coeff, basis)
            raw = raw.index_select(-1, self.nonself_index_tensor)
        else:
            raw = self.phase_raw
        values = math.pi * torch.tanh(raw)
        return omega.index_copy(-1, self.nonself_index_tensor, values)

    def lambdas(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._alpha()
        omega = self._omega()
        real_lambda = torch.complex(alpha, torch.zeros_like(alpha))
        complex_lambda = torch.complex(alpha, omega)
        carry_lambda = (
            complex_lambda
            if self.phase_route in {"both", "carry_only"}
            else real_lambda
        )
        correction_lambda = (
            complex_lambda
            if self.phase_route in {"both", "correction_only"}
            else real_lambda
        )
        return (carry_lambda, correction_lambda)

    def _mix(self) -> torch.Tensor:
        imaginary = torch.zeros_like(self.mix_real)
        imaginary = imaginary.index_copy(-1, self.nonself_index_tensor, self.mix_imag)
        return _project_mixer(torch.complex(self.mix_real, imaginary), self.channels)

    def _injection_budget(self) -> torch.Tensor:
        radius = self._radius(self.inj_bias.device, self.inj_bias.dtype)
        return 2.0 * torch.sigmoid(
            self.inj_bias - F.softplus(self.inj_slope_raw) * radius
        )

    def forward(
        self,
        v: torch.Tensor,
        g: Optional[torch.Tensor],
        *,
        runtime_flags: BranchFlags,
        return_components: bool = False,
    ):
        if v.ndim != 3 or v.shape[1] != self.channels or v.shape[-1] != self.num_points:
            raise ValueError(
                f"expected B,{self.channels},{self.num_points}; got {tuple(v.shape)}"
            )
        carry_on = runtime_flags.carry and self.support_carry
        correction_on = runtime_flags.correction and self.support_correction
        if correction_on and g is None:
            raise ValueError("correction input is required when correction is active")
        v_hat = torch.fft.rfft(v.float(), dim=-1)
        low_hat = torch.zeros_like(v_hat[..., : self.modes])
        carry_lambda, correction_lambda = self.lambdas()
        components: Dict[str, torch.Tensor] = {}
        if carry_on:
            carry_hat = torch.exp(carry_lambda).unsqueeze(0) * v_hat[..., : self.modes]
            low_hat = low_hat + carry_hat
            if return_components:
                components["carry_hat"] = carry_hat
        if correction_on:
            assert g is not None
            g_hat = torch.fft.rfft(g.float(), dim=-1)[..., : self.modes]
            mixed = torch.einsum("bim,iom->bom", g_hat, self._mix())
            forcing_hat = (
                stable_phi1(correction_lambda).unsqueeze(0)
                * mixed
                * self._injection_budget().unsqueeze(0)
            )
            low_hat = low_hat + forcing_hat
            if return_components:
                components["correction_hat"] = forcing_hat
        mask = self.imaginary_mode_mask.to(
            device=low_hat.device, dtype=low_hat.real.dtype
        )
        low_hat = torch.complex(low_hat.real, low_hat.imag * mask)
        tail = low_hat.new_zeros((*low_hat.shape[:-1], v_hat.shape[-1] - self.modes))
        out_hat = torch.cat((low_hat, tail), dim=-1)
        result = torch.fft.irfft(out_hat, n=self.num_points, dim=-1)
        return (result, components) if return_components else result

    @torch.no_grad()
    def generator_audit(self) -> Dict[str, float]:
        carry_lambda, correction_lambda = self.lambdas()
        phi = stable_phi1(correction_lambda)
        omega = carry_lambda.imag
        return {
            "alpha_min": float(carry_lambda.real.min().item()),
            "alpha_max": float(carry_lambda.real.max().item()),
            "omega_abs_max": float(omega.abs().max().item()),
            "carry_modulus_max": float(torch.exp(carry_lambda).abs().max().item()),
            "phi1_modulus_max": float(phi.abs().max().item()),
        }


class CorrectedSGNO1dCore(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_points: int,
        *,
        modes: int = 26,
        width: int = 11,
        n_blocks: int = 7,
        generator_type: str = "real",
        branch_variant: str = "full",
        phase_route: str = "both",
        decay_init: str = "canonical",
        coordinate_mode: str = "none",
    ):
        super().__init__()
        if coordinate_mode not in {"none", "raw", "periodic"}:
            raise ValueError("coordinate_mode must be none, raw, or periodic")
        self.num_channels = int(num_channels)
        self.num_points = int(num_points)
        self.modes = int(modes)
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.generator_type = generator_type
        self.branch_variant = branch_variant
        self.phase_route = phase_route
        self.decay_init = decay_init
        self.coordinate_mode = coordinate_mode
        self.architecture_flags = branch_flags(branch_variant)
        self.runtime_branch_variant: Optional[str] = None
        coord_channels = {"none": 0, "raw": 1, "periodic": 2}[coordinate_mode]
        self.lift = nn.Conv1d(num_channels + coord_channels, width, 1)
        self.corrections = nn.ModuleList()
        self.generators = nn.ModuleList()
        self.bypasses = nn.ModuleList()
        for _ in range(n_blocks):
            if self.architecture_flags.correction:
                self.corrections.append(PointwiseCorrection1d(width))
            else:
                self.corrections.append(nn.Identity())
            if self.architecture_flags.carry or self.architecture_flags.correction:
                self.generators.append(
                    CorrectedSpectralGenerator1d(
                        width,
                        modes,
                        num_points,
                        generator_type=generator_type,
                        phase_route=phase_route,
                        support_carry=self.architecture_flags.carry,
                        support_correction=self.architecture_flags.correction,
                        decay_init=decay_init,
                    )
                )
            else:
                self.generators.append(nn.Identity())
            self.bypasses.append(
                nn.Conv1d(width, width, 1)
                if self.architecture_flags.bypass
                else nn.Identity()
            )
        self.head1 = nn.Conv1d(width, HEAD_WIDTH, 1)
        self.head2 = nn.Conv1d(HEAD_WIDTH, num_channels, 1)

    def _coordinates(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.coordinate_mode == "none":
            return None
        coord = (
            torch.arange(self.num_points, device=x.device, dtype=x.dtype)
            / self.num_points
        )
        coord = coord.view(1, 1, -1).expand(x.shape[0], 1, -1)
        if self.coordinate_mode == "raw":
            return coord
        angle = 2.0 * math.pi * coord
        return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)

    def _runtime_flags(self) -> BranchFlags:
        if self.runtime_branch_variant is None:
            return self.architecture_flags
        requested = branch_flags(self.runtime_branch_variant)
        return BranchFlags(
            requested.carry and self.architecture_flags.carry,
            requested.correction and self.architecture_flags.correction,
            requested.bypass and self.architecture_flags.bypass,
        )

    def set_runtime_branch_variant(self, variant: Optional[str]) -> None:
        if variant is not None:
            branch_flags(variant)
        self.runtime_branch_variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = self._coordinates(x)
        if coordinates is not None:
            x = torch.cat((x, coordinates), dim=1)
        x = self.lift(x)
        flags = self._runtime_flags()
        for block_index in range(self.n_blocks):
            terms = []
            if flags.carry or flags.correction:
                correction = (
                    self.corrections[block_index](x) if flags.correction else None
                )
                terms.append(
                    self.generators[block_index](x, correction, runtime_flags=flags)
                )
            if flags.bypass:
                terms.append(self.bypasses[block_index](x))
            if not terms:
                raise RuntimeError("runtime knockout removed every branch")
            update = terms[0]
            for term in terms[1:]:
                update = update + term
            x = update if block_index == self.n_blocks - 1 else F.gelu(update)
        return self.head2(F.gelu(self.head1(x)))

    @torch.no_grad()
    def generator_audit(self) -> Dict[str, object]:
        audits = [
            module.generator_audit()
            for module in self.generators
            if isinstance(module, CorrectedSpectralGenerator1d)
        ]
        if not audits:
            return {"num_generator_blocks": 0}
        keys = audits[0]
        return {
            "num_generator_blocks": len(audits),
            **{
                f"{key}_across_blocks": {
                    "min": min((audit[key] for audit in audits)),
                    "max": max((audit[key] for audit in audits)),
                }
                for key in keys
            },
        }


class OneStepWrapper1d(nn.Module):
    def __init__(self, core: nn.Module, *, outer_residual: bool):
        super().__init__()
        self.core = core
        self.outer_residual = bool(outer_residual)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        delta_or_state = self.core(u)
        return u + delta_or_state if self.outer_residual else delta_or_state

    def set_runtime_branch_variant(self, variant: Optional[str]) -> None:
        setter = getattr(self.core, "set_runtime_branch_variant", None)
        if setter is None:
            if variant is not None:
                raise ValueError("this model does not support branch knockouts")
            return
        setter(variant)

    @torch.no_grad()
    def generator_audit(self) -> Dict[str, object]:
        audit = getattr(self.core, "generator_audit", None)
        return audit() if audit is not None else {"num_generator_blocks": 0}


class ApeBenchSpectralConv1d(nn.Module):
    """PyTorch port of PDEquinox v0.1.2's raw-parameter 1D spectral conv."""

    def __init__(
        self, in_channels: int, out_channels: int, modes: int, num_points: int
    ):
        super().__init__()
        available = num_points // 2 + 1
        if modes > available:
            raise ValueError(f"modes={modes} exceeds {available} rFFT bins")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        self.num_points = int(num_points)
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = torch.fft.rfft(x.float(), dim=-1)
        weight = torch.complex(self.weight_real, self.weight_imag)
        low = torch.einsum("bim,iom->bom", x_hat[..., : self.modes], weight)
        out_hat = x_hat.new_zeros((x.shape[0], self.out_channels, x_hat.shape[-1]))
        out_hat[..., : self.modes] = low
        return torch.fft.irfft(out_hat, n=self.num_points, dim=-1)


class ApeBenchFNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int, num_points: int):
        super().__init__()
        self.spectral = ApeBenchSpectralConv1d(width, width, modes, num_points)
        self.bypass = nn.Conv1d(width, width, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.bypass(x))


class ApeBenchFNO1d(nn.Module):
    """Topology-compatible PyTorch port of pinned APEBench ClassicFNO."""

    def __init__(
        self,
        num_channels: int,
        num_points: int,
        *,
        width: int = 14,
        modes: int = 33,
        n_blocks: int = 4,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_points = int(num_points)
        self.width = int(width)
        self.modes = int(modes)
        self.n_blocks = int(n_blocks)
        self.lift = nn.Conv1d(num_channels, width, 1, bias=True)
        self.blocks = nn.ModuleList(
            [ApeBenchFNOBlock1d(width, modes, num_points) for _ in range(n_blocks)]
        )
        self.project = nn.Conv1d(width, num_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)


def build_corrected_sgno_1d(
    *,
    num_channels: int,
    num_points: int,
    width: int,
    modes: int,
    n_blocks: int,
    generator_type: str,
    branch_variant: str,
    phase_route: str,
    decay_init: str,
    coordinate_mode: str,
    outer_residual: bool,
) -> OneStepWrapper1d:
    core = CorrectedSGNO1dCore(
        num_channels,
        num_points,
        width=width,
        modes=modes,
        n_blocks=n_blocks,
        generator_type=generator_type,
        branch_variant=branch_variant,
        phase_route=phase_route,
        decay_init=decay_init,
        coordinate_mode=coordinate_mode,
    )
    return OneStepWrapper1d(core, outer_residual=outer_residual)


def build_apebench_fno_1d(
    *,
    num_channels: int,
    num_points: int,
    width: int,
    modes: int,
    n_blocks: int,
    outer_residual: bool,
) -> OneStepWrapper1d:
    core = ApeBenchFNO1d(
        num_channels, num_points, width=width, modes=modes, n_blocks=n_blocks
    )
    return OneStepWrapper1d(core, outer_residual=outer_residual)


def expected_apebench_fno_real_dof(
    *, num_channels: int, width: int, modes: int, n_blocks: int
) -> int:
    return (
        n_blocks * ((2 * modes + 1) * width * width + width)
        + 2 * num_channels * width
        + width
        + num_channels
    )


def aggregate_generator_audit(model: nn.Module) -> Dict[str, object]:
    audit = getattr(model, "generator_audit", None)
    return audit() if audit is not None else {"num_generator_blocks": 0}


def iter_corrected_generators(
    model: nn.Module,
) -> Iterable[CorrectedSpectralGenerator1d]:
    for module in model.modules():
        if isinstance(module, CorrectedSpectralGenerator1d):
            yield module
