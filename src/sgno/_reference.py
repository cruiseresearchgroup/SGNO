"""Construct the dimension-specific SGNO backbone used by the paper."""

from __future__ import annotations
from sgno import build_sgno_from_config
from sgno.sgno import _parse_network_config

CURRENT_PROFILE = "sgno_2d_init_small_20260914"
LEGACY_PROFILE = "sgno_canonical"
CURRENT_NETWORK = "sgno_canonical;width=5;modes=10;n_blocks=9;initial_step=1"


def model_profile_metadata(config: dict) -> dict[str, str]:
    profile = config.get("model_profile", LEGACY_PROFILE)
    if profile == LEGACY_PROFILE:
        return {
            "model_profile": LEGACY_PROFILE,
            "core_version": "legacy_public",
            "mixer_initialization": "legacy_dimension_rule",
        }
    if profile != CURRENT_PROFILE:
        raise ValueError(f"unknown model_profile {profile!r}")
    for key, expected in (("spatial_dim", 2), ("num_channels", 1), ("num_points", 64)):
        value = config.get(key)
        if isinstance(value, bool) or value != expected:
            raise ValueError(f"{CURRENT_PROFILE} requires {key}={expected}")
    architecture = _parse_network_config(config.get("network_config", ""))
    if architecture != dict(width=5, modes=10, n_blocks=9, initial_step=1):
        raise ValueError(
            f"{CURRENT_PROFILE} requires network_config={CURRENT_NETWORK!r}"
        )
    from sgno._frequency import SGNO_SIGNED_RADIUS_2D_VERSION

    return {
        "model_profile": CURRENT_PROFILE,
        "core_version": SGNO_SIGNED_RADIUS_2D_VERSION,
        "mixer_initialization": "matched_1d_rule_private_rng_v1",
    }


def build_from_run_config(config: dict, *, seed: int):
    metadata = model_profile_metadata(config)
    if metadata["model_profile"] == CURRENT_PROFILE:
        from sgno._initialization import build_variant

        model = build_variant("init_small", seed=seed)
    else:
        model = build_sgno_from_config(
            str(config["network_config"]),
            int(config["spatial_dim"]),
            int(config.get("num_points", 0)),
            int(config["num_channels"]),
        )
    model.model_profile = metadata["model_profile"]
    return model


def assert_checkpoint_profile(checkpoint: dict, config: dict) -> None:
    expected = model_profile_metadata(config)
    actual_profile = checkpoint.get("model_profile", LEGACY_PROFILE)
    if actual_profile != expected["model_profile"]:
        raise ValueError("checkpoint/config mismatch for model_profile")
    for key in ("core_version", "mixer_initialization"):
        if key not in checkpoint and actual_profile == LEGACY_PROFILE:
            continue
        if checkpoint.get(key) != expected[key]:
            raise ValueError(f"checkpoint/config mismatch for {key}")
