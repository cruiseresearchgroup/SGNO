"""Generate SGNO paper-task trajectories under an auditable data contract.

Canonical generation is intentionally strict: it requires the exact APEBench
checkout, fixed task parameters, the CPU backend, and JAX float32 execution.
Any supported resolution or test-set override is a named non-canonical variant.
"""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "false"
import numpy as np
import jax
import jax.numpy as jnp


def _jax_x64_enabled() -> bool:
    if hasattr(jax.config, "x64_enabled"):
        return bool(jax.config.x64_enabled)
    return bool(jax.config.read("jax_enable_x64"))


def _assert_jax_runtime() -> None:
    backend = jax.default_backend()
    x64_enabled = _jax_x64_enabled()
    default_dtype = str(jnp.asarray(0.0).dtype)
    if backend != "cpu" or x64_enabled or default_dtype != "float32":
        raise RuntimeError(
            f"canonical data generation requires JAX CPU float32 execution; got backend={backend!r}, x64_enabled={x64_enabled}, default_dtype={default_dtype!r}"
        )


_assert_jax_runtime()
import apebench
from apebench.scenarios import difficulty, physical

EXPECTED_APEBENCH_COMMIT = "dca24a00eeacbe5524a99542f6d5c2bc2deb8da8"
EXPECTED_APEBENCH_VERSION = "0.1.1"
PROVENANCE_CATEGORIES = (
    "paper-explicit",
    "public-config-inferred",
    "APEBench-default-inferred",
    "user-variant",
)


@dataclass(frozen=True)
class TaskSpec:
    scenario_class: type
    num_spatial_dims: int
    num_points: int
    paper_dynamic: str
    num_points_source: str
    num_points_detail: str
    canonical_num_test_samples: int = 30
    canonical_num_warmup_steps: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)
    canonical_num_channels: int = 1
    canonical_ic_config: str = "fourier;5;true;true"
    task_source_category: str = "paper-explicit"
    task_source_detail: str = (
        "This task is included in the paper comparison."
    )


TASK_SPECS = {
    "adv1d": TaskSpec(
        difficulty.Advection,
        1,
        160,
        "adv",
        "APEBench-default-inferred",
        "N=160 matches the pinned APEBench v0.1.1 BaseScenario default.",
    ),
    "diff1d": TaskSpec(
        difficulty.Diffusion,
        1,
        160,
        "diff",
        "APEBench-default-inferred",
        "N=160 matches the pinned APEBench v0.1.1 BaseScenario default.",
    ),
    "disp1d": TaskSpec(
        difficulty.Dispersion,
        1,
        160,
        "disp",
        "APEBench-default-inferred",
        "N=160 matches the pinned APEBench v0.1.1 BaseScenario default.",
    ),
    "kdv1d": TaskSpec(
        difficulty.KortewegDeVries,
        1,
        160,
        "kdv",
        "APEBench-default-inferred",
        "N=160 matches the pinned APEBench v0.1.1 BaseScenario default.",
    ),
    "ks1d": TaskSpec(
        difficulty.KuramotoSivashinsky,
        1,
        160,
        "ks",
        "APEBench-default-inferred",
        "N=160 matches the pinned APEBench v0.1.1 BaseScenario default.",
        canonical_num_warmup_steps=500,
    ),
    "mixdisp2d": TaskSpec(
        physical.SpatiallyMixedDispersion,
        2,
        64,
        "mix_disp",
        "public-config-inferred",
        "N=64 is the paper model grid; it overrides the pinned scenario default.",
    ),
    "kolm2d": TaskSpec(
        physical.KolmogorovFlow,
        2,
        64,
        "kolm_flow",
        "paper-explicit",
        "The Kolmogorov model uses a 64x64 grid.",
        canonical_num_warmup_steps=500,
    ),
    "diagdiff3d": TaskSpec(
        physical.DiagonalDiffusion,
        3,
        32,
        "diag_diff",
        "public-config-inferred",
        "N=32 matches the paper configuration and pinned APEBench 3D curation.",
        canonical_num_test_samples=10,
        kwargs={
            "diffusion_coef_vector": (0.001, 0.002, 0.0004),
            "num_test_samples": 10,
        },
    ),
    "unbaladv3d": TaskSpec(
        physical.UnbalancedAdvection,
        3,
        32,
        "unbal_adv",
        "public-config-inferred",
        "N=32 matches the paper configuration and pinned APEBench 3D curation.",
        canonical_num_test_samples=10,
        kwargs={"num_test_samples": 10},
    ),
    "advdiff3d": TaskSpec(
        difficulty.AdvectionDiffusion,
        3,
        32,
        "adv_diff",
        "public-config-inferred",
        "N=32 matches the paper configuration and pinned APEBench 3D curation.",
        canonical_num_test_samples=10,
        kwargs={"num_test_samples": 10},
    ),
}
SCENARIO_ATTRIBUTES = (
    "num_spatial_dims",
    "num_points",
    "num_channels",
    "domain_extent",
    "dt",
    "ic_config",
    "num_warmup_steps",
    "num_train_samples",
    "train_temporal_horizon",
    "train_seed",
    "num_test_samples",
    "test_temporal_horizon",
    "test_seed",
    "optim_config",
    "batch_size",
    "advection_coef",
    "diffusion_coef",
    "dipersion_coef",
    "dispersion_coef",
    "hyper_diffusion_coef",
    "advection_gamma",
    "diffusion_gamma",
    "dispersion_gamma",
    "hyp_diffusion_gamma",
    "convection_sc_delta",
    "gradient_norm_delta",
    "gammas",
    "deltas",
    "a_coefs",
    "advection_coef_vector",
    "diffusion_coef_vector",
    "diffusivity",
    "injection_mode",
    "injection_scale",
    "drag",
    "num_substeps",
    "order",
    "dealiasing_fraction",
    "num_circle_points",
    "circle_radius",
    "coarse_proportion",
    "maximum_absolute",
    "vlim",
    "report_metrics",
    "diffusion_coef_matrix",
    "diffusivity_1",
    "diffusivity_2",
    "feed_rate",
    "kill_rate",
    "pattern_type",
    "reactivity",
    "critical_number",
    "polynomial_coefficients",
)
SOLVER_ATTRIBUTES = (
    "num_spatial_dims",
    "domain_extent",
    "num_points",
    "num_channels",
    "dt",
    "order",
    "num_sub_steps",
    "linear_difficulties",
    "nonlinear_difficulties",
    "normalized_linear_coefficients",
    "normalized_nonlinear_coefficients",
    "linear_coefficients",
    "velocity",
    "diffusivity",
    "dispersivity",
    "advect_on_diffusion",
    "convection_scale",
    "vorticity_convection_scale",
    "drag",
    "injection_mode",
    "injection_scale",
    "dealiasing_fraction",
    "num_circle_points",
    "circle_radius",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        if array.size <= 256:
            return array.tolist()
        return {"shape": list(array.shape), "dtype": str(array.dtype)}
    return repr(value)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _locate_apebench_repo(explicit_repo: Path | None) -> Path | None:
    if explicit_repo is not None:
        return explicit_repo.expanduser().resolve()
    imported_path = Path(apebench.__file__).resolve()
    for candidate in (imported_path.parent, *imported_path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _inspect_apebench_checkout(explicit_repo: Path | None) -> dict[str, Any]:
    imported_path = Path(apebench.__file__).resolve()
    status: dict[str, Any] = {
        "verified": False,
        "expected_commit": EXPECTED_APEBENCH_COMMIT,
        "expected_version": EXPECTED_APEBENCH_VERSION,
        "imported_from": str(imported_path),
        "requested_repo": str(explicit_repo) if explicit_repo is not None else None,
    }
    repo = _locate_apebench_repo(explicit_repo)
    if repo is None:
        status["reason"] = (
            "could not locate a git checkout containing the imported apebench package"
        )
        return status
    try:
        root = Path(_run_git(repo, "rev-parse", "--show-toplevel")).resolve()
        head = _run_git(root, "rev-parse", "HEAD").lower()
        tracked_status = _run_git(
            root, "status", "--porcelain", "--untracked-files=no", "--", "apebench"
        )
    except (OSError, subprocess.CalledProcessError) as error:
        status["repo"] = str(repo)
        status["reason"] = f"git verification failed: {error}"
        return status
    version = _package_version("apebench")
    package_root = (root / "apebench").resolve()
    reasons = []
    if not _path_is_within(imported_path, package_root):
        reasons.append(f"imported package is outside verified checkout {package_root}")
    if head != EXPECTED_APEBENCH_COMMIT:
        reasons.append(f"HEAD is {head}, expected {EXPECTED_APEBENCH_COMMIT}")
    if tracked_status:
        reasons.append("tracked files under apebench/ differ from HEAD")
    if version != EXPECTED_APEBENCH_VERSION:
        reasons.append(
            f"package version is {version}, expected {EXPECTED_APEBENCH_VERSION}"
        )
    status.update(
        {
            "repo": str(root),
            "head": head,
            "package_version": version,
            "tracked_source_clean": not bool(tracked_status),
            "verified": not reasons,
            "reason": "; ".join(reasons) if reasons else None,
        }
    )
    return status


def _verify_apebench_checkout(
    explicit_repo: Path | None, *, required: bool
) -> dict[str, Any]:
    status = _inspect_apebench_checkout(explicit_repo)
    if required and (not status["verified"]):
        raise RuntimeError(
            f"canonical generation requires the exact clean APEBench checkout {EXPECTED_APEBENCH_COMMIT}; {status.get('reason')}. Pass --apebench-repo PATH to the checkout used by this Python environment."
        )
    return status


def _resolve_generation_mode(
    num_points_override: int | None,
    num_test_samples_override: int | None,
    variant_descriptor: str | None,
) -> tuple[bool, str | None, list[str]]:
    overrides = []
    if num_points_override is not None:
        overrides.append("num_points")
    if num_test_samples_override is not None:
        overrides.append("num_test_samples")
    descriptor = variant_descriptor.strip() if variant_descriptor is not None else None
    if descriptor == "":
        raise ValueError("--variant-descriptor must not be empty")
    if overrides and descriptor is None:
        raise ValueError(
            "parameter overrides are non-canonical and require an explicit --variant-descriptor"
        )
    canonical = not overrides and descriptor is None
    return (canonical, descriptor, overrides)


def _different_override(value: int | None, canonical_value: int) -> int | None:
    """Return only a value that actually differs from the canonical contract."""
    if value is None or value == canonical_value:
        return None
    return value


def _validate_output_path(path: Path) -> Path:
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "--out must end in .npz so the saved and hashed paths are identical"
        )
    return path


def _canonical_contract(task: str, spec: TaskSpec) -> dict[str, Any]:
    return {
        "task_alias": task,
        "num_spatial_dims": spec.num_spatial_dims,
        "num_points": spec.num_points,
        "num_channels": spec.canonical_num_channels,
        "ic_config": spec.canonical_ic_config,
        "num_warmup_steps": spec.canonical_num_warmup_steps,
        "num_train_samples": 50,
        "train_temporal_horizon": 50,
        "train_seed": 0,
        "num_test_samples": spec.canonical_num_test_samples,
        "test_temporal_horizon": 200,
        "test_seed": 773,
        "train_shape": [50, 51, spec.canonical_num_channels]
        + [spec.num_points] * spec.num_spatial_dims,
        "test_shape": [
            spec.canonical_num_test_samples,
            201,
            spec.canonical_num_channels,
        ]
        + [spec.num_points] * spec.num_spatial_dims,
        "dtype": "float32",
    }


def _assert_canonical_scenario(task: str, scenario: Any, spec: TaskSpec) -> None:
    contract = _canonical_contract(task, spec)
    checked_fields = (
        "num_spatial_dims",
        "num_points",
        "num_channels",
        "ic_config",
        "num_warmup_steps",
        "num_train_samples",
        "train_temporal_horizon",
        "train_seed",
        "num_test_samples",
        "test_temporal_horizon",
        "test_seed",
    )
    mismatches = []
    for name in checked_fields:
        actual = getattr(scenario, name, "<missing>")
        if actual != contract[name]:
            mismatches.append(f"{name}={actual!r} (expected {contract[name]!r})")
    if mismatches:
        raise RuntimeError(
            f"canonical scenario contract failed for {task}: " + ", ".join(mismatches)
        )


def _as_float32_trajectory(value: Any, *, split: str) -> np.ndarray:
    source_dtype = str(getattr(value, "dtype", "<missing>"))
    if source_dtype != "float32":
        raise RuntimeError(
            f"{split} trajectory was computed as {source_dtype}, expected JAX float32 before NumPy conversion"
        )
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise RuntimeError(f"{split} NumPy dtype is {array.dtype}, expected float32")
    return array


def _describe_solver(stepper: Any) -> dict[str, Any]:
    description = {
        "class": f"{type(stepper).__module__}.{type(stepper).__name__}",
        "parameters": {
            name: _json_value(getattr(stepper, name))
            for name in SOLVER_ATTRIBUTES
            if hasattr(stepper, name)
        },
    }
    nested = getattr(stepper, "stepper", None)
    if nested is not None and nested is not stepper:
        description["nested_stepper"] = _describe_solver(nested)
    return description


def _field_provenance(task: str, spec: TaskSpec) -> dict[str, Any]:
    return {
        "task_dynamic": {
            "category": spec.task_source_category,
            "detail": spec.task_source_detail,
        },
        "scenario_class": {
            "category": "APEBench-default-inferred",
            "detail": "The exact Python scenario class is selected from the pinned APEBench v0.1.1 registry/curation.",
        },
        "num_points": {
            "category": spec.num_points_source,
            "detail": spec.num_points_detail,
        },
        "test_temporal_horizon": {
            "category": "paper-explicit",
            "detail": "The paper explicitly specifies a 200-step closed-loop evaluation.",
        },
        "train_protocol_and_numeric_data_seeds": {
            "category": "APEBench-default-inferred",
            "detail": "50 training trajectories, horizon 50, train seed 0, test seed 773, and default test counts come from the pinned APEBench scenario/curation.",
        },
        "num_test_samples": {
            "category": "APEBench-default-inferred",
            "detail": f"The canonical test count {spec.canonical_num_test_samples} comes from the pinned APEBench default/3D curation.",
        },
    }


def _effective_contract(
    scenario: Any, train: np.ndarray, test: np.ndarray
) -> dict[str, Any]:
    return {
        "num_spatial_dims": scenario.num_spatial_dims,
        "num_points": scenario.num_points,
        "num_channels": scenario.num_channels,
        "ic_config": scenario.ic_config,
        "num_warmup_steps": scenario.num_warmup_steps,
        "num_train_samples": scenario.num_train_samples,
        "train_temporal_horizon": scenario.train_temporal_horizon,
        "train_seed": scenario.train_seed,
        "num_test_samples": scenario.num_test_samples,
        "test_temporal_horizon": scenario.test_temporal_horizon,
        "test_seed": scenario.test_seed,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_dtype": str(train.dtype),
        "test_dtype": str(test.dtype),
    }


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": _jax_x64_enabled(),
        "jax_default_dtype": str(jnp.asarray(0.0).dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--num-points", type=int)
    parser.add_argument("--num-test-samples", type=int)
    parser.add_argument("--variant-descriptor")
    parser.add_argument("--apebench-repo", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = TASK_SPECS[args.task]
    num_points_override = _different_override(args.num_points, spec.num_points)
    num_test_samples_override = _different_override(
        args.num_test_samples, spec.canonical_num_test_samples
    )
    try:
        canonical, variant_descriptor, overridden_fields = _resolve_generation_mode(
            num_points_override, num_test_samples_override, args.variant_descriptor
        )
        out_path = _validate_output_path(args.out)
    except ValueError as error:
        parser.error(str(error))
    apebench_source = _verify_apebench_checkout(args.apebench_repo, required=canonical)
    kwargs = dict(spec.kwargs)
    kwargs["num_spatial_dims"] = spec.num_spatial_dims
    kwargs["num_points"] = (
        spec.num_points if args.num_points is None else args.num_points
    )
    if args.num_test_samples is not None:
        kwargs["num_test_samples"] = args.num_test_samples
    scenario = spec.scenario_class(**kwargs)
    if canonical:
        _assert_canonical_scenario(args.task, scenario, spec)
    ref_stepper = scenario.get_ref_stepper()
    train = _as_float32_trajectory(scenario.get_train_data(), split="train")
    test = _as_float32_trajectory(scenario.get_test_data(), split="test")
    spatial = (scenario.num_points,) * scenario.num_spatial_dims
    expected_train = (
        scenario.num_train_samples,
        scenario.train_temporal_horizon + 1,
        scenario.num_channels,
        *spatial,
    )
    expected_test = (
        scenario.num_test_samples,
        scenario.test_temporal_horizon + 1,
        scenario.num_channels,
        *spatial,
    )
    if train.shape != expected_train or test.shape != expected_test:
        raise RuntimeError(
            f"unexpected shapes train={train.shape}, test={test.shape}, expected {expected_train}, {expected_test}"
        )
    if canonical:
        contract = _canonical_contract(args.task, spec)
        if (
            list(train.shape) != contract["train_shape"]
            or list(test.shape) != contract["test_shape"]
        ):
            raise RuntimeError(
                f"canonical saved-shape contract failed: train={train.shape}, test={test.shape}, contract={contract['train_shape']}/{contract['test_shape']}"
            )
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise RuntimeError("APEBench generated non-finite data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, train=train, test=test)
    field_provenance = _field_provenance(args.task, spec)
    if num_points_override is not None:
        field_provenance["num_points"] = {
            "category": "user-variant",
            "detail": f"Effective N={scenario.num_points} is the named variant override; canonical N={spec.num_points} has source {spec.num_points_source}.",
        }
    if num_test_samples_override is not None:
        field_provenance["num_test_samples"] = {
            "category": "user-variant",
            "detail": f"Effective test count {scenario.num_test_samples} is the named variant override; canonical count is {spec.canonical_num_test_samples}.",
        }
    metadata = {
        "task_alias": args.task,
        "canonical": canonical,
        "variant_descriptor": variant_descriptor,
        "overridden_fields": overridden_fields,
        "provenance_categories": list(PROVENANCE_CATEGORIES),
        "field_provenance": field_provenance,
        "canonical_contract": _canonical_contract(args.task, spec),
        "effective_contract": _effective_contract(scenario, train, test),
        "scenario_name": scenario.get_scenario_name(),
        "scenario_class": f"{spec.scenario_class.__module__}.{spec.scenario_class.__name__}",
        "scenario_constructor_kwargs": _json_value(kwargs),
        "scenario_config": {
            name: _json_value(getattr(scenario, name))
            for name in SCENARIO_ATTRIBUTES
            if hasattr(scenario, name)
        },
        "reference_solver": _describe_solver(ref_stepper),
        "apebench_source": apebench_source,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_dtype": str(train.dtype),
        "test_dtype": str(test.dtype),
        "train_sha256": _array_sha256(train),
        "test_sha256": _array_sha256(test),
        "npz_path": str(out_path.resolve()),
        "npz_sha256": _file_sha256(out_path),
        "versions": {
            name: _package_version(name)
            for name in (
                "apebench",
                "jax",
                "jaxlib",
                "equinox",
                "optax",
                "exponax",
                "pdequinox",
                "trainax",
                "numpy",
            )
        },
        "runtime": _runtime_metadata(),
        "generation_backend": "APEBench reference get_train_data/get_test_data; verified JAX CPU float32",
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"DATA={out_path}")
    print(f"METADATA={metadata_path}")


if __name__ == "__main__":
    main()
