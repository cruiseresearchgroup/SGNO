from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sgno import build_sgno_from_config
from sgno.data import load_trajectory_splits


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _initial_step_from_config(network_config: str) -> int:
    for part in network_config.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().lower() == "initial_step":
            return int(float(value.strip()))
    return 1


def _learning_rate(
    update_index: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak_lr: float,
) -> float:
    """Optax-compatible linear warmup followed by cosine decay."""

    if update_index < warmup_updates:
        return float(peak_lr) * update_index / max(1, warmup_updates)
    decay_updates = max(1, total_updates - warmup_updates)
    progress = min(1.0, (update_index - warmup_updates) / decay_updates)
    return float(peak_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class TrajDataset(Dataset):
    def __init__(self, trajectories: np.ndarray, initial_step: int):
        self.u = np.asarray(trajectories).astype(np.float32, copy=False)
        self.initial_step = int(initial_step)
        if self.u.ndim < 4:
            raise ValueError("trajectory array must be at least 4D")
        self.num_trajectories = int(self.u.shape[0])
        self.num_times = int(self.u.shape[1])
        if self.num_times <= self.initial_step:
            raise ValueError("T must be larger than initial_step")

    def __len__(self) -> int:
        return self.num_trajectories * (self.num_times - self.initial_step)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = index // (self.num_times - self.initial_step)
        start = index % (self.num_times - self.initial_step)
        stop = start + self.initial_step
        x = self.u[trajectory, start:stop]
        y = self.u[trajectory, stop]
        return torch.from_numpy(x), torch.from_numpy(y)


def _resolve_training_value(cli_value, train_config: dict, key: str, default):
    return cli_value if cli_value is not None else train_config.get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--train-fraction", type=float)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    spatial_dim = int(config["spatial_dim"])
    num_channels = int(config["num_channels"])
    num_points = int(config.get("num_points", 0))
    network_config = str(config["network_config"])
    train_config = dict(config.get("train", {}))

    if "epochs" in train_config and "updates" not in train_config and args.updates is None:
        raise ValueError(
            "epoch-based training is not the paper protocol; replace train.epochs "
            "with train.updates (10,000 for the reported setting)"
        )

    updates = int(_resolve_training_value(args.updates, train_config, "updates", 10_000))
    warmup_updates = int(
        _resolve_training_value(args.warmup_updates, train_config, "warmup_updates", 2_000)
    )
    batch_size = int(_resolve_training_value(args.batch_size, train_config, "batch_size", 20))
    peak_lr = float(_resolve_training_value(args.lr, train_config, "lr", 1.0e-3))
    seed = int(_resolve_training_value(args.seed, train_config, "seed", 0))
    log_every = int(_resolve_training_value(args.log_every, train_config, "log_every", 100))
    train_fraction = float(
        _resolve_training_value(args.train_fraction, train_config, "train_fraction", 0.8)
    )

    if updates < 1 or batch_size < 1 or log_every < 1:
        raise ValueError("updates, batch_size, and log_every must be positive")
    if not 0 <= warmup_updates < updates:
        raise ValueError("warmup_updates must satisfy 0 <= warmup_updates < updates")
    if peak_lr <= 0:
        raise ValueError("lr must be positive")

    _set_seed(seed)
    train, _val, _test, split_metadata = load_trajectory_splits(
        args.data,
        train_fraction=train_fraction,
    )
    expected_ndim = spatial_dim + 3
    if train.ndim != expected_ndim:
        raise ValueError(
            f"expected training trajectory rank {expected_ndim} for {spatial_dim}D data, "
            f"got shape {train.shape}"
        )
    if int(train.shape[2]) != num_channels:
        raise ValueError(f"expected {num_channels} training channels, got {int(train.shape[2])}")
    if not np.isfinite(train).all():
        raise ValueError("training trajectories contain non-finite values")

    initial_step = _initial_step_from_config(network_config)
    dataset = TrajDataset(train, initial_step=initial_step)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=shuffle_generator,
    )
    iterator = iter(loader)

    device = torch.device(args.device)
    model = build_sgno_from_config(
        network_config,
        spatial_dim,
        num_points,
        num_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr)
    loss_fn = nn.MSELoss()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"final_update_{updates}.pt"
    state_path = output_dir / "state.json"

    model.train()
    last_loss = math.nan
    last_lr = math.nan
    for update_index in range(updates):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        x = x.to(device)
        y = y.to(device)
        last_lr = _learning_rate(
            update_index,
            total_updates=updates,
            warmup_updates=warmup_updates,
            peak_lr=peak_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = last_lr

        prediction = model(x)
        loss = loss_fn(prediction, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        completed_updates = update_index + 1
        if completed_updates % log_every == 0 or completed_updates == updates:
            _write_json(
                state_path,
                {
                    "completed_updates": completed_updates,
                    "last_train_loss": last_loss,
                    "last_learning_rate": last_lr,
                    "checkpoint_policy": "final update only; no validation/test selection",
                    "test_data_used_during_training": False,
                },
            )

    checkpoint = {
        "format_version": 2,
        "model_variant": "sgno_canonical",
        "coordinate_mode": "legacy_raw_inclusive",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "network_config": network_config,
        "spatial_dim": spatial_dim,
        "num_channels": num_channels,
        "num_points": num_points,
        "data_split": split_metadata,
        "training_protocol": {
            "objective": "one-step MSE",
            "optimizer": "Adam",
            "updates": updates,
            "warmup_updates": warmup_updates,
            "schedule": "linear warmup then cosine decay",
            "peak_lr": peak_lr,
            "batch_size": batch_size,
            "seed": seed,
            "optimizer_updates": updates,
            "checkpoint_policy": "final update only; no validation/test selection",
        },
    }
    torch.save(checkpoint, checkpoint_path)
    print(os.fspath(checkpoint_path))


if __name__ == "__main__":
    main()
