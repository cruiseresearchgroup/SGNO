from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from sgno.data import load_trajectory_split, load_trajectory_splits


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_array_split_is_replayed_for_evaluation(tmp_path):
    trajectories = np.arange(10 * 3 * 1 * 4, dtype=np.float32).reshape(10, 3, 1, 4)
    data_path = tmp_path / "single.npz"
    np.savez(data_path, u=trajectories)

    train, val, test, metadata = load_trajectory_splits(data_path, train_fraction=0.8)
    assert val is None
    assert np.array_equal(train, trajectories[:8])
    assert np.array_equal(test, trajectories[8:])

    selected, selected_metadata = load_trajectory_split(
        data_path,
        split="test",
        split_metadata=metadata,
    )
    assert np.array_equal(selected, trajectories[8:])
    assert selected_metadata["selected_count"] == 2


def test_explicit_splits_are_isolated_and_mixed_schema_is_rejected(tmp_path):
    train = np.zeros((3, 3, 1, 4), dtype=np.float32)
    val = np.ones((2, 3, 1, 4), dtype=np.float32)
    test = np.full((4, 3, 1, 4), 2.0, dtype=np.float32)
    data_path = tmp_path / "explicit.npz"
    np.savez(data_path, train=train, val=val, test=test)

    selected, _ = load_trajectory_split(data_path, split="val")
    assert np.array_equal(selected, val)

    ambiguous_path = tmp_path / "ambiguous.npz"
    np.savez(ambiguous_path, train=train, test=test, u=np.concatenate([train, test]))
    with pytest.raises(ValueError, match="cannot mix"):
        load_trajectory_splits(ambiguous_path)


def test_learning_rate_matches_warmup_cosine_endpoints():
    train_module = _load_script_module("sgno_public_train", "scripts/train_npz.py")
    learning_rate = train_module._learning_rate
    assert learning_rate(0, total_updates=10_000, warmup_updates=2_000, peak_lr=1.0e-3) == 0.0
    assert learning_rate(2_000, total_updates=10_000, warmup_updates=2_000, peak_lr=1.0e-3) == pytest.approx(1.0e-3)
    assert learning_rate(10_000, total_updates=10_000, warmup_updates=2_000, peak_lr=1.0e-3) == pytest.approx(0.0)


def _run_public_training(config_path: Path, data_path: Path, output_dir: Path) -> dict:
    environment = os.environ.copy()
    source_path = str(REPO_ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_npz.py"),
            "--config",
            str(config_path),
            "--data",
            str(data_path),
            "--out",
            str(output_dir),
            "--device",
            "cpu",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
        timeout=60,
        capture_output=True,
        text=True,
    )
    checkpoint_path = output_dir / "final_update_3.pt"
    assert checkpoint_path.exists()
    assert not (output_dir / "best.pt").exists()
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def test_test_contents_do_not_change_training_checkpoint_and_short_eval_is_labeled(tmp_path):
    rng = np.random.default_rng(7)
    train = rng.standard_normal((4, 4, 1, 8), dtype=np.float32)
    test_a = rng.standard_normal((2, 4, 1, 8), dtype=np.float32)
    test_b = np.full_like(test_a, 123.0)
    data_a = tmp_path / "data_a.npz"
    data_b = tmp_path / "data_b.npz"
    np.savez(data_a, train=train, test=test_a)
    np.savez(data_b, train=train, test=test_b)

    config = {
        "spatial_dim": 1,
        "num_channels": 1,
        "num_points": 8,
        "network_config": "sgno_canonical;width=2;modes=2;n_blocks=1;initial_step=1",
        "train": {
            "batch_size": 2,
            "updates": 3,
            "warmup_updates": 1,
            "lr": 0.001,
            "log_every": 1,
            "seed": 0,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    checkpoint_a = _run_public_training(config_path, data_a, tmp_path / "run_a")
    checkpoint_b = _run_public_training(config_path, data_b, tmp_path / "run_b")
    for key, value in checkpoint_a["model"].items():
        assert torch.equal(value, checkpoint_b["model"][key])

    state = json.loads((tmp_path / "run_a" / "state.json").read_text(encoding="utf-8"))
    assert state["completed_updates"] == 3
    assert state["test_data_used_during_training"] is False
    assert "best_val" not in state and "val_loss" not in state

    evaluation_path = tmp_path / "eval.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_npz.py"),
            "--config",
            str(config_path),
            "--data",
            str(data_a),
            "--ckpt",
            str(tmp_path / "run_a" / "final_update_3.pt"),
            "--out",
            str(evaluation_path),
            "--steps",
            "3",
            "--device",
            "cpu",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
        timeout=60,
        capture_output=True,
        text=True,
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert evaluation["split"] == "test"
    assert evaluation["n_traj"] == 2
    assert evaluation["gmean100"] is None
    assert evaluation["prefix_length"] == 3
    assert evaluation["tail_length"] == 0
