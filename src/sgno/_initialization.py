"""Seeded complex-mixer initialization for the two-dimensional backbone."""

from __future__ import annotations
import math
from pathlib import Path
import sys
from typing import Any
import torch
from torch import nn
from sgno._frequency import SGNO_SIGNED_RADIUS_2D_VERSION, build_regime_sgno
from sgno.sgno import SGNOTorchApeWrapper

VARIANT_NAMES = ("baseline", "init_small", "wide", "wide_init_small")
MIXER_SEED_OFFSET = 104729
_SEED_MODULUS = 2**63
_REFERENCE_REAL_PARAMETERS = 101402
_NARROW_CONFIG = {"width": 5, "modes": 10, "n_blocks": 9}
_WIDE_CONFIG = {"width": 11, "modes": 10, "n_blocks": 2}


def _validate(name: str, seed: int, ndim: int) -> None:
    if name not in VARIANT_NAMES:
        raise ValueError(
            f"unknown 2D tuning variant {name!r}; expected {VARIANT_NAMES}"
        )
    if isinstance(ndim, bool) or ndim != 2:
        raise ValueError("this isolated configuration experiment supports only ndim=2")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or (not 0 <= seed < _SEED_MODULUS)
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")


@torch.no_grad()
def _initialize_mixers_small(model: SGNOTorchApeWrapper, seed: int) -> None:
    """Preserve parameter identities and all non-mixer state/RNG exactly."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + MIXER_SEED_OFFSET) % _SEED_MODULUS)
    for block in model.core.etds:
        scale = 1.0 / (block.channels * block.channels)
        for name in ("mix_pos", "mix_neg"):
            parameter = getattr(block, name)
            values = torch.rand(
                parameter.shape,
                dtype=parameter.dtype,
                device="cpu",
                generator=generator,
            )
            parameter.copy_(scale * values)


def build_variant(name: str, seed: int, ndim: int = 2) -> SGNOTorchApeWrapper:
    """Build a raw B,1,X,Y -> B,1,X,Y wrapper; leave optimizer policy unchanged.

    ``baseline`` and ``wide`` retain the complete original 2D initialization.
    Their paired ``*_small`` controls only redraw complex mixers after full
    construction, using the 1D distribution/scale rule, rand(C,C,M,M) / C**2.
    """
    _validate(name, seed, ndim)
    config = _WIDE_CONFIG if name.startswith("wide") else _NARROW_CONFIG
    torch.manual_seed(seed)
    with torch.device("cpu"):
        model = build_regime_sgno(
            num_spatial_dims=2, num_channels=1, initial_step=1, **config
        )
    if name in ("init_small", "wide_init_small"):
        _initialize_mixers_small(model, seed)
    model.tuning_variant_name = name
    model.tuning_seed = seed
    return model
