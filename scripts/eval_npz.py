from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sgno import build_sgno_from_config
from sgno.data import load_trajectory_split


def _initial_step_from_config(network_config: str) -> int:
    for part in network_config.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().lower() == "initial_step":
            return int(float(value.strip()))
    return 1


def _nrmse(prediction: torch.Tensor, reference: torch.Tensor, eps: float = 1.0e-12) -> float:
    numerator = torch.linalg.norm(prediction - reference)
    denominator = torch.linalg.norm(reference) + eps
    return float((numerator / denominator).item())


def _gmean(values: list[float] | np.ndarray, cap: float = 0.0, eps: float = 1.0e-12) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return math.nan
    if cap > 0:
        array = np.minimum(array, float(cap))
    return float(np.exp(np.mean(np.log(array + float(eps)))))


def _stable_step(values: list[float] | np.ndarray, threshold: float) -> tuple[int, bool]:
    array = np.asarray(values, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(array) | (array > float(threshold)))
    if bad.size:
        return int(bad[0] + 1), False
    return int(array.size), True


def _first_nonfinite_step(values: list[float] | np.ndarray) -> int | None:
    bad = np.flatnonzero(~np.isfinite(np.asarray(values, dtype=np.float64)))
    return int(bad[0] + 1) if bad.size else None


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _assert_checkpoint_matches_config(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    expected = {
        "network_config": str(config["network_config"]),
        "spatial_dim": int(config["spatial_dim"]),
        "num_channels": int(config["num_channels"]),
    }
    for key, value in expected.items():
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing required metadata {key!r}")
        if checkpoint[key] != value:
            raise ValueError(
                f"checkpoint/config mismatch for {key}: "
                f"checkpoint={checkpoint[key]!r}, config={value!r}"
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--cap", type=float, default=0.0)
    parser.add_argument("--max-traj", type=int, default=0)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.max_traj < 0:
        raise ValueError("max_traj cannot be negative")

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    checkpoint = _load_checkpoint(args.ckpt)
    _assert_checkpoint_matches_config(checkpoint, config)

    spatial_dim = int(config["spatial_dim"])
    num_channels = int(config["num_channels"])
    num_points = int(config.get("num_points", 0))
    network_config = str(config["network_config"])
    initial_step = _initial_step_from_config(network_config)

    trajectories, selected_split = load_trajectory_split(
        args.data,
        split=args.split,
        split_metadata=checkpoint.get("data_split"),
        train_fraction=args.train_fraction,
    )
    trajectories = np.asarray(trajectories).astype(np.float32, copy=False)
    expected_ndim = spatial_dim + 3
    if trajectories.ndim != expected_ndim:
        raise ValueError(
            f"expected trajectory rank {expected_ndim} for {spatial_dim}D data, "
            f"got shape {trajectories.shape}"
        )
    if int(trajectories.shape[2]) != num_channels:
        raise ValueError(
            f"expected {num_channels} channels, got {int(trajectories.shape[2])}"
        )
    if not np.isfinite(trajectories).all():
        raise ValueError("evaluation trajectories contain non-finite values")

    num_trajectories = int(trajectories.shape[0])
    if args.max_traj > 0:
        num_trajectories = min(num_trajectories, int(args.max_traj))
    if num_trajectories < 1:
        raise ValueError("selected evaluation split is empty")

    total_times = int(trajectories.shape[1])
    horizon = min(int(args.steps), total_times - initial_step)
    if horizon < 1:
        raise ValueError("trajectory does not contain a prediction step after the history window")

    device = torch.device(args.device)
    model = build_sgno_from_config(
        network_config,
        spatial_dim,
        num_points,
        num_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    u = torch.from_numpy(trajectories)
    per_trajectory: list[dict[str, Any]] = []
    all_errors: list[float] = []
    step_errors: list[list[float]] = [[] for _ in range(horizon)]

    with torch.no_grad():
        for trajectory_index in range(num_trajectories):
            history = u[trajectory_index, :initial_step].unsqueeze(0).to(device)
            errors: list[float] = []
            for step in range(horizon):
                prediction = model(history)
                reference = u[trajectory_index, initial_step + step].to(device)
                error = _nrmse(prediction, reference)
                errors.append(error)
                step_errors[step].append(error)
                if initial_step == 1:
                    history = prediction.unsqueeze(1)
                else:
                    history = torch.cat([history[:, 1:], prediction.unsqueeze(1)], dim=1)

            stable_step, stable_through_horizon = _stable_step(errors, args.tau)
            prefix_length = min(100, horizon)
            tail_length = max(0, min(200, horizon) - 100)
            per_trajectory.append(
                {
                    "gmean100": _gmean(errors[:100], args.cap) if horizon >= 100 else None,
                    "gmean_prefix": _gmean(errors[:prefix_length], args.cap),
                    "prefix_length": prefix_length,
                    "tail_gmean_101_200": _gmean(errors[100:200], args.cap)
                    if horizon > 100
                    else None,
                    "tail_length": tail_length,
                    "stable_step": stable_step,
                    "stable_through_horizon": stable_through_horizon,
                    "final_nrmse": float(errors[-1]),
                    "max_nrmse": float(np.max(np.asarray(errors, dtype=np.float64))),
                    "first_nonfinite_step": _first_nonfinite_step(errors),
                }
            )
            all_errors.extend(errors)

    mean_nrmse_by_step = [
        float(np.mean(np.asarray(values, dtype=np.float64))) for values in step_errors
    ]
    stable_step, stable_through_horizon = _stable_step(mean_nrmse_by_step, args.tau)
    prefix_length = min(100, horizon)
    tail_length = max(0, min(200, horizon) - 100)
    trajectory_gmeans = np.asarray(
        [item["gmean_prefix"] for item in per_trajectory],
        dtype=np.float64,
    )
    trajectory_stable_steps = np.asarray(
        [item["stable_step"] for item in per_trajectory],
        dtype=np.float64,
    )

    output = {
        "split": args.split,
        "split_metadata": selected_split,
        "n_traj": num_trajectories,
        "horizon": horizon,
        "tau": float(args.tau),
        "cap": float(args.cap),
        "gmean100": _gmean(mean_nrmse_by_step[:100], args.cap) if horizon >= 100 else None,
        "gmean_prefix": _gmean(mean_nrmse_by_step[:prefix_length], args.cap),
        "prefix_length": prefix_length,
        "tail_gmean_101_200": _gmean(mean_nrmse_by_step[100:200], args.cap)
        if horizon > 100
        else None,
        "tail_length": tail_length,
        "stable_step": stable_step,
        "stable_through_horizon": stable_through_horizon,
        "all_steps_finite": bool(np.isfinite(np.asarray(mean_nrmse_by_step)).all()),
        "first_nonfinite_step": _first_nonfinite_step(mean_nrmse_by_step),
        "final_mean_nrmse": float(mean_nrmse_by_step[-1]),
        "max_mean_nrmse": float(np.max(np.asarray(mean_nrmse_by_step, dtype=np.float64))),
        "mean_nrmse_by_step": mean_nrmse_by_step,
        "mean_nrmse": float(np.mean(np.asarray(all_errors, dtype=np.float64))),
        "median_nrmse": float(np.median(np.asarray(all_errors, dtype=np.float64))),
        "median_traj_gmean_prefix": float(np.median(trajectory_gmeans)),
        "mean_traj_gmean_prefix": float(np.mean(trajectory_gmeans)),
        "median_traj_stable_step": float(np.median(trajectory_stable_steps)),
        "q25_traj_stable_step": float(np.quantile(trajectory_stable_steps, 0.25)),
        "q75_traj_stable_step": float(np.quantile(trajectory_stable_steps, 0.75)),
        "per_traj": per_trajectory,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(output), handle, indent=2, sort_keys=True, allow_nan=False)
    print(output_path)


if __name__ == "__main__":
    main()

