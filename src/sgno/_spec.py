"""Paper grid sizes, architectures, and backbone construction."""

from pathlib import Path
import sys
import torch
from torch import nn
from sgno._reference import CURRENT_PROFILE, LEGACY_PROFILE, build_from_run_config

METHOD = "sgno_state_propagation_v1"
VERSION = "full-grid-state-skip-axis-quadratic-decay-v1-20260917"
SPATIAL_SIZES = {1: 160, 2: 64, 3: 32}
ARCHITECTURES = {1: (11, 26, 7), 2: (5, 10, 9), 3: (10, 6, 2)}
BASE_COUNTS = {1: (30573, 52595), 2: (56402, 101402), 3: (192827, 365627)}
INITIAL_AXIS_AMPLITUDE = 0.1


def build_reference(ndim, seed=0):
    width, modes, n_blocks = ARCHITECTURES[ndim]
    torch.manual_seed(seed)
    return build_from_run_config(
        dict(
            spatial_dim=ndim,
            num_channels=1,
            num_points=SPATIAL_SIZES[ndim],
            model_profile=CURRENT_PROFILE if ndim == 2 else LEGACY_PROFILE,
            network_config=f"sgno_canonical;width={width};modes={modes};n_blocks={n_blocks};initial_step=1",
        ),
        seed=seed,
    )
