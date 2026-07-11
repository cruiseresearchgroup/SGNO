from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _validate_train_fraction(train_fraction: float) -> float:
    value = float(train_fraction)
    if not 0.0 < value < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    return value


def _split_single_array(
    u: np.ndarray,
    *,
    train_fraction: float,
    train_end: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if u.ndim < 4:
        raise ValueError("trajectory array must be at least 4D")
    if u.shape[0] < 2:
        raise ValueError("the single-array NPZ schema requires at least two trajectories")

    if train_end is None:
        fraction = _validate_train_fraction(train_fraction)
        train_end = int(fraction * int(u.shape[0]))
    train_end = int(train_end)
    if not 1 <= train_end < int(u.shape[0]):
        raise ValueError("single-array split must leave at least one train and one test trajectory")
    return u[:train_end], u[train_end:], train_end


def load_trajectory_splits(
    path: str | Path,
    *,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Load train/validation/test trajectories without reusing the test split.

    Supported schemas are ``train`` + optional ``val`` + ``test``, or one
    array named ``u``.  The latter is split deterministically along the first
    axis.  The returned metadata is checkpoint-safe and lets evaluation replay
    exactly the same split.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        keys = set(data.files)
        explicit_keys = keys.intersection({"train", "val", "test"})
        if "u" in keys and explicit_keys:
            raise ValueError("NPZ cannot mix the single-array and explicit split schemas")
        if "train" in keys or "test" in keys or "val" in keys:
            if not {"train", "test"}.issubset(keys):
                raise ValueError("explicit NPZ schema requires both train and test arrays")
            train = np.asarray(data["train"])
            val = np.asarray(data["val"]) if "val" in keys else None
            test = np.asarray(data["test"])
            metadata: dict[str, Any] = {
                "schema": "explicit",
                "train_count": int(train.shape[0]),
                "val_count": int(val.shape[0]) if val is not None else 0,
                "test_count": int(test.shape[0]),
            }
            return train, val, test, metadata

        if "u" in keys:
            u = np.asarray(data["u"])
            train, test, train_end = _split_single_array(
                u,
                train_fraction=train_fraction,
            )
            metadata = {
                "schema": "u_split",
                "total_count": int(u.shape[0]),
                "train_end": int(train_end),
                "train_fraction": float(train_fraction),
                "train_count": int(train.shape[0]),
                "val_count": 0,
                "test_count": int(test.shape[0]),
            }
            return train, None, test, metadata

    raise ValueError("NPZ must contain train and test arrays, or one array named u")


def load_trajectory_split(
    path: str | Path,
    *,
    split: str = "test",
    split_metadata: dict[str, Any] | None = None,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one named split, replaying a checkpoint's ``u`` split if present.

    ``split='all'`` is intentionally opt-in.  Evaluation defaults to ``test``
    so a single-array NPZ can never silently mix training trajectories into a
    reported test metric.
    """

    split = str(split).lower()
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be one of train, val, test, or all")

    with np.load(Path(path), allow_pickle=False) as data:
        keys = set(data.files)
        explicit_keys = keys.intersection({"train", "val", "test"})
        if "u" in keys and explicit_keys:
            raise ValueError("NPZ cannot mix the single-array and explicit split schemas")
        if "train" in keys or "test" in keys or "val" in keys:
            if not {"train", "test"}.issubset(keys):
                raise ValueError("explicit NPZ schema requires both train and test arrays")
            arrays = {
                name: np.asarray(data[name])
                for name in ("train", "val", "test")
                if name in keys
            }
            if split == "all":
                selected = np.concatenate([arrays[name] for name in ("train", "val", "test") if name in arrays])
            elif split in arrays:
                selected = arrays[split]
            else:
                raise ValueError(f"NPZ has no {split!r} split")
            return selected, {
                "schema": "explicit",
                "selected_split": split,
                "selected_count": int(selected.shape[0]),
            }

        if "u" in keys:
            u = np.asarray(data["u"])
            checkpoint_split = split_metadata or {}
            train_end = None
            if checkpoint_split.get("schema") == "u_split":
                expected_total = int(checkpoint_split.get("total_count", -1))
                if expected_total != int(u.shape[0]):
                    raise ValueError(
                        "checkpoint split metadata does not match the NPZ trajectory count"
                    )
                train_end = int(checkpoint_split["train_end"])
            train, test, train_end = _split_single_array(
                u,
                train_fraction=train_fraction,
                train_end=train_end,
            )
            if split == "train":
                selected = train
            elif split == "test":
                selected = test
            elif split == "all":
                selected = u
            else:
                raise ValueError("single-array NPZ schema has no validation split")
            return selected, {
                "schema": "u_split",
                "selected_split": split,
                "selected_count": int(selected.shape[0]),
                "total_count": int(u.shape[0]),
                "train_end": int(train_end),
            }

    raise ValueError("NPZ must contain train and test arrays, or one array named u")
