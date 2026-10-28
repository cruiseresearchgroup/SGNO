"""Aggregate complete seed groups without dropping non-finite runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    for f in args.runs.rglob("summary.json"):
        r = json.loads(f.read_text())
        if r.get("smoke_only"):
            continue
        key = (r["task"], r["corrector"], r["outer"], r.get("component", "full"))
        groups[key].append(r)
    output = []
    for key, rows in sorted(groups.items()):
        if sorted(r["seed"] for r in rows) != list(range(5)):
            raise ValueError(f"{key}: expected exactly seeds 0 through 4")
        rows.sort(key=lambda r: r["seed"])
        if len({r["steps"] for r in rows}) != 1:
            raise ValueError(f"{key}: prediction horizons must match")
        metrics = [
            "gmean_1_100",
            "gmean_101_200",
            "endpoint_nrmse",
            "late_absolute_slope",
            "late_log_slope",
        ]
        metrics += ["endpoint_" + name for name in (
            "amplitude_squared", "phase_squared", "total_squared",
            "squared_norm_ratio", "field_correlation")]
        for metric in metrics:
            present = [metric in r for r in rows]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"{key}: {metric} is missing in some runs")
            values = np.asarray([float(r[metric]) for r in rows])
            finite = np.isfinite(values)
            output.append(
                dict(
                    task=key[0],
                    corrector=key[1],
                    outer=key[2],
                    component=key[3],
                    steps=rows[0]["steps"],
                    metric=metric,
                    median=float(np.median(values)) if finite.all() else "NF",
                    mean=float(values.mean()) if finite.all() else "NF",
                    sample_sd=float(values.std(ddof=1)) if finite.all() else "NF",
                    nonfinite_seeds=int((~finite).sum()),
                    **{f"run_{chr(65+i)}": float(v) if np.isfinite(v) else "NF"
                       for i, v in enumerate(values)},
                )
            )
    if not output:
        raise ValueError("No complete benchmark evaluations found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(output[0]))
        w.writeheader()
        w.writerows(output)
    print(f"Wrote {len(output)} summaries to {args.out}")


if __name__ == "__main__":
    main()
