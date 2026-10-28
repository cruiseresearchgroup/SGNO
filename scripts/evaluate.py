"""Evaluate a saved predictor recursively without reference resets."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sgno.ablations import build_predictor
from sgno.metrics import relative_rms, summarize, json_safe, field_diagnostics
from sgno.components import apply as apply_component


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    if any((args.out / name).exists() for name in ("metrics.npz", "summary.json", "sample_fields.npz")):
        raise FileExistsError("Choose a new output directory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["settings"]
    if saved["step"] != cfg["updates"]:
        raise ValueError("Use the fixed final checkpoint")
    with np.load(args.data, allow_pickle=False) as z:
        test = np.ascontiguousarray(z["test"][:, : args.steps + 1])
    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/tasks.json").read_text()
    )[cfg["task"]]
    if test.dtype != np.float32 or test.shape[2:] != tuple(spec["test_shape"][2:]):
        raise ValueError("Test channel/grid shape mismatch")
    if len(test) != spec["test_shape"][0] and not cfg["smoke_only"]:
        raise ValueError("Evaluate all test initial conditions")
    if test.shape[1] != args.steps + 1:
        raise ValueError("Test horizon is too short")
    model = build_predictor(cfg["ndim"], cfg["seed"], cfg["corrector"], cfg["outer"])
    if cfg.get("component", "full") != "full":
        apply_component(model, cfg["component"])
    model.load_state_dict(saved["model"], strict=True)
    model.to(args.device).eval()
    errors = np.empty((len(test), args.steps), np.float64)
    endpoint = {}
    sample = [test[0, 0].copy()]
    batch = 10 if cfg["ndim"] == 3 else 5
    for start in range(0, len(test), batch):
        current = torch.from_numpy(test[start : start + batch, 0]).to(args.device)
        for t in range(1, args.steps + 1):
            current = model(current)
            value = current.cpu().numpy()
            errors[start : start + len(value), t - 1] = relative_rms(
                value, test[start : start + len(value), t], cfg["ndim"]
            )
            if start == 0:
                sample.append(value[0].copy())
        diagnostics = field_diagnostics(value, test[start : start + len(value), args.steps])
        for key, values in diagnostics.items():
            endpoint.setdefault(key, []).append(values)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "metrics.npz").exists():
        raise FileExistsError("Choose a new output directory")
    endpoint = {"endpoint_" + key: np.concatenate(values) for key, values in endpoint.items()}
    np.savez_compressed(args.out / "metrics.npz", nrmse=errors, **endpoint)
    np.savez_compressed(
        args.out / "sample_fields.npz", prediction=np.stack(sample), reference=test[0]
    )
    summary = dict(
        summarize(errors),
        **{key: float(values.mean()) for key, values in endpoint.items()},
        task=cfg["task"],
        seed=cfg["seed"],
        corrector=cfg["corrector"],
        outer=cfg["outer"],
        component=cfg.get("component", "full"),
        smoke_only=cfg["smoke_only"],
    )
    (args.out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2))
    print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
