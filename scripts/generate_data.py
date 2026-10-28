"""Generate a paper task with the pinned original numerical protocol."""

import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--apebench-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / "configs/tasks.json").read_text())
    if args.task not in tasks:
        p.error("Unknown paper task")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generators = root / "src/sgno/data_generation"
    if args.task not in ("burgers1d_nu12e4", "anisodiff2d_axis", "anisodiff2d_rot45"):
        sys.argv = [
            "anchors.py",
            "--task",
            args.task,
            "--apebench-repo",
            str(args.apebench_repo),
            "--out",
            str(args.out),
        ]
        runpy.run_path(str(generators / "anchors.py"), run_name="__main__")
        return
    spec = importlib.util.spec_from_file_location(
        "physical_data", generators / "physical.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    jax, ex, physical, provenance = module.load_runtime(args.apebench_repo)
    report = module.run_qa(args.task, jax, ex, physical, provenance)
    qa_path = args.out.with_suffix(".qa.json")
    module.write_json_exclusive(qa_path, report)
    if report["status"] != "PASS":
        raise RuntimeError("Reference QA failed; see " + str(qa_path))
    module.generate(
        args.task,
        args.out,
        qa_path,
        module.object_hash(module.contract(args.task)),
        jax,
        ex,
        physical,
        provenance,
    )


if __name__ == "__main__":
    main()
