"""Train a fixed final-checkpoint predictor using adjacent training states."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sgno.ablations import build_predictor
from sgno.components import apply as apply_component


def learning_rate(step, updates=10000, warmup=2000, peak=1e-3):
    if step < warmup:
        return peak * step / max(1, warmup)
    return (
        peak
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, updates - warmup))))
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, choices=range(5), default=0)
    p.add_argument("--corrector", choices=["sgno", "fno"], default="sgno")
    p.add_argument(
        "--outer",
        choices=[
            "fitted_joint",
            "identity_fixed",
            "direct_identity",
            "direct_fitted",
            "direct_fitted_frozen",
        ],
        default="fitted_joint",
    )
    p.add_argument(
        "--component",
        choices=["full", "no_nonlinear_spectral_update", "no_local_pointwise_path"],
        default="full",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--smoke-updates",
        type=int,
        default=0,
        help="Short software check; never a paper result.",
    )
    args = p.parse_args()
    tasks = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/tasks.json").read_text()
    )
    if args.task not in tasks:
        p.error("Unknown paper task")
    spec = tasks[args.task]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = args.outer == "fitted_joint"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Training reads the train key only; evaluation is a separate command.
    with np.load(args.data, allow_pickle=False) as z:
        train = np.ascontiguousarray(z["train"])
    if train.dtype != np.float32 or train.shape[2:] != tuple(spec["train_shape"][2:]):
        raise ValueError("Data must be float32 with the configured channel/grid shape")
    if not args.smoke_updates and list(train.shape) != spec["train_shape"]:
        raise ValueError(
            "Paper training requires all 50 trajectories and 50 adjacent transitions"
        )
    if train.shape[1] < 2 or not np.isfinite(train).all():
        raise ValueError("Invalid training trajectories")
    model = build_predictor(spec["ndim"], args.seed, args.corrector, args.outer)
    initialization = model.initialize_from_train(train)
    if args.component != "full":
        apply_component(model, args.component)
    model.to(args.device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    settings = dict(
        task=args.task,
        ndim=spec["ndim"],
        seed=args.seed,
        corrector=args.corrector,
        outer=args.outer,
        component=args.component,
        updates=args.smoke_updates or 10000,
        warmup=2000,
        batch_size=20,
        peak_lr=0.001,
        smoke_only=bool(args.smoke_updates),
        train_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "checkpoint.pt"
    sampler = torch.Generator(device="cpu").manual_seed(args.seed)
    step, cursor, permutation = 0, 0, None
    if checkpoint.exists():
        if not args.resume:
            raise FileExistsError("Existing checkpoint: use --resume or another --out")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["settings"] != settings:
            raise ValueError("Resume settings/data mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.ndim:
                    state[k] = v.to(args.device)
        sampler.set_state(saved["sampler"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda") and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, cursor, permutation = saved["step"], saved["cursor"], saved["permutation"]
    else:
        (args.out / "config.json").write_text(json.dumps(settings, indent=2))
        (args.out / "initialization.json").write_text(
            json.dumps(initialization, indent=2)
        )
    transitions = train.shape[1] - 1
    count = len(train) * transitions
    array = torch.from_numpy(train).to(args.device)
    while step < settings["updates"]:
        if permutation is None:
            permutation = torch.randperm(count, generator=sampler)
            cursor = 0
        ids = permutation[cursor : cursor + 20].to(args.device)
        current, target = (
            array[ids // transitions, ids % transitions],
            array[ids // transitions, ids % transitions + 1],
        )
        rate = learning_rate(step, settings["updates"], settings["warmup"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        prediction = model(current)
        loss = (
            (prediction - target).square().mean()
            if spec["ndim"] == 3
            else torch.nn.functional.mse_loss(prediction, target)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {step + 1}")
        loss.backward()
        if not all(
            torch.isfinite(v.grad).all()
            for v in model.parameters()
            if v.grad is not None
        ):
            raise FloatingPointError(f"Non-finite gradient at update {step + 1}")
        optimizer.step()
        if hasattr(model, "project_constraints"):
            model.project_constraints()
        step += 1
        cursor += len(ids)
        if cursor >= count:
            permutation = None
            cursor = 0
        if step == 1 or step % 100 == 0 or step == settings["updates"]:
            row = {"update": step, "loss": float(loss.detach()), "learning_rate": rate}
            with (args.out / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step % 500 == 0 or step == settings["updates"]:
            temp = checkpoint.with_suffix(".tmp")
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    settings=settings,
                    step=step,
                    cursor=cursor,
                    permutation=permutation,
                    sampler=sampler.get_state(),
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()
                    if args.device.startswith("cuda")
                    else None,
                ),
                temp,
            )
            temp.replace(checkpoint)


if __name__ == "__main__":
    main()
