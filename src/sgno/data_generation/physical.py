"""Isolated physical-regime data and reference QA; never edits old registries.

``describe`` needs only NumPy. ``audit`` imports, but does not integrate, the
verified solver. Run ``qa`` on allocated CPU resources before ``generate``.
Generation requires a matching PASS report and exact contract hash. Numerical
failures do not trigger grid, coefficient, integrator, or tolerance adjustment.

The numerical settings below are an explicit new-batch reference policy, NOT
APEBench defaults. Source equations are RHS conventions: c/2 * d_x(u**2), since
APEBench supplies scale=-c and Exponax supplies -scale/2. This avoids both the
common sign error and a factor-of-two error in an expanded Burgers equation.
"""

from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np

APEBENCH_COMMIT = "dca24a00eeacbe5524a99542f6d5c2bc2deb8da8"
PINNED_VERSIONS = {
    "apebench": "0.1.1",
    "exponax": "0.1.0",
    "jax": "0.4.35",
    "jaxlib": "0.4.34",
    "equinox": "0.11.10",
    "numpy": "1.24.2",
    "optax": "0.2.4",
    "pdequinox": "0.1.2",
    "trainax": "0.0.2",
    "scipy": "1.15.3",
}
EX_SOURCE_HASHES = {
    "nonlin_fun/_convection.py": "0deaf2df2de07d915cda546c7b89232c0c0413d13811621572821f8f41f9767f",
    "stepper/generic/_convection.py": "2dcac920649662536bc6801f91f56531c54960891f21df2aea676a3fae32e0b2",
    "stepper/_diffusion.py": "5536229721a8faa256c3b5044eb74b07813432bfe3425f528a1872fbaccc60d2",
}
SPLITS = {
    "train": {"seed": 91001, "samples": 50, "horizon": 50},
    "validation": {"seed": 91002, "samples": 30, "horizon": 200},
    "test": {"seed": 91003, "samples": 30, "horizon": 1000},
}
QA_POLICY = {
    "seed": 91004,
    "samples": 3,
    "horizon": 20,
    "temporal_substep_multipliers": [1, 2, 4],
    "spatial_grid_multiplier": 2,
    "time_coarse_max_relative_rms": 0.0001,
    "time_fine_max_relative_rms": 3e-05,
    "grid_max_relative_rms": 0.01,
    "dealias_max_relative_rms": 0.01,
    "high_grid_unresolved_energy_fraction": 0.001,
    "mass_max_absolute_drift": 1e-09,
    "analytic_max_relative_rms": 1e-09,
    "float32_max_relative_rms": 1e-06,
    "relative_rms_floor": 1e-12,
    "time_convergence_ratio_max": 0.8,
    "time_convergence_ratio_noise_floor": 1e-10,
    "scope": "Independent fixed ICs; bounded short-time reference checks, not model-score selection or a global error theorem.",
    "cks_warmup_scope": "One N640 reference warmup of 500 output steps; all short-time solver comparisons restart from that identical full N640 warmed state. No chaotic long-horizon pointwise convergence claim.",
}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    family: str
    scenario_class: str
    ndim: int
    n: int
    length: float
    dt: float = 0.1
    convection: float | None = None
    diffusion: float | None = None
    hyperdiffusion: float = 0.0
    warmup: int = 0
    matrix: tuple[tuple[float, float], tuple[float, float]] | None = None


TASKS = {
    "burgers1d_nu12e4": TaskSpec(
        "burgers", "Burgers", 1, 160, 1.0, diffusion=0.0012, convection=-0.125
    ),
    "anisodiff2d_axis": TaskSpec(
        "anisotropic_diffusion",
        "AnisotropicDiffusion",
        2,
        64,
        1.0,
        matrix=((0.001, 0.0), (0.0, 0.002)),
    ),
    "anisodiff2d_rot45": TaskSpec(
        "anisotropic_diffusion",
        "AnisotropicDiffusion",
        2,
        64,
        1.0,
        matrix=((0.0015, -0.0005), (-0.0005, 0.0015)),
    ),
}
SOLVER_POLICY = {
    "integration_dtype": "float64",
    "storage_dtype": "float32",
    "backend": "cpu",
    "nonlinear_order": 4,
    "nonlinear_substeps": 10,
    "dealiasing_fraction": 2 / 3,
    "num_circle_points": 32,
    "circle_radius": 1.0,
    "linear_order": 0,
    "linear_substeps": 1,
    "nonlinear_reference_points": 640,
    "note": "Explicit new-reference settings, not the pinned scenario defaults (order2/substeps1/contour16).",
}


def grid_contract(task: str) -> dict[str, Any]:
    """The numerical reference grid is not the model's observation grid."""
    spec = TASKS[task]
    reference_n = (
        SOLVER_POLICY["nonlinear_reference_points"] if spec.matrix is None else spec.n
    )
    return {
        "observation_points_per_axis": spec.n,
        "reference_points_per_axis": reference_n,
        "qa_refined_points_per_axis": reference_n * QA_POLICY["spatial_grid_multiplier"]
        if spec.matrix is None
        else None,
        "initial_sampling_points_per_axis": spec.n,
        "initial_prolongation": "Repeated periodic Fourier interpolation with even-grid Nyquist splitting"
        if reference_n != spec.n
        else "identity",
        "restriction": "aligned_point_subsampling",
        "restriction_stride": reference_n // spec.n,
        "restriction_description": "Evaluate at the original left-endpoint-included observation-grid coordinates; no averaging or low-pass filter.",
        "mass_qa_grid": "full_reference_grid",
        "float32_rounding_qa_grid": "observation_grid",
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def effective_coefficients(spec: TaskSpec) -> dict[str, Any]:
    if spec.matrix is not None:
        matrix = np.asarray(spec.matrix, dtype=np.float64)
        if (
            not np.array_equal(matrix, matrix.T)
            or np.linalg.eigvalsh(matrix).min() <= 0
        ):
            raise ValueError("Diffusion tensor must be symmetric positive definite")
        return {
            "pde_rhs": "sum_ij A_ij * d_i d_j u",
            "diffusion_tensor": matrix.tolist(),
            "eigenvalues": np.linalg.eigvalsh(matrix).tolist(),
            "mixed_derivative_coefficient": float(2 * matrix[0, 1]),
            "fourier_symbol": "-k^T A k, k_j=2*pi*n_j/L",
            "boundary": "periodic, left endpoint included, right endpoint excluded",
        }
    return {
        "pde_rhs": "(c/2)*d_x(u^2) + a2*d_xx(u) + a4*d_xxxx(u)",
        "apebench_convection_coef_c": spec.convection,
        "exponax_convection_scale": -spec.convection,
        "rhs_derivative_of_square_coefficient": spec.convection / 2,
        "rhs_u_times_derivative_coefficient": spec.convection,
        "linear_derivative_coefficients": [
            0.0,
            0.0,
            spec.diffusion,
            0.0,
            spec.hyperdiffusion,
        ],
        "conservative": True,
        "single_channel": False,
        "boundary": "periodic, left endpoint included, right endpoint excluded",
        "fourier_symbol": "-a2*k^2+a4*k^4, k=2*pi*n/L",
    }


def contract(task: str) -> dict[str, Any]:
    spec = TASKS[task]
    return {
        "schema": "sgno-regime-data-v1",
        "task_alias": task,
        "task": dataclasses.asdict(spec),
        "effective_coefficients": effective_coefficients(spec),
        "grid": grid_contract(task),
        "ic_config": "fourier;5;true;true",
        "num_channels": 1,
        "warmup_output_steps": spec.warmup,
        "warmup_physical_time": spec.warmup * spec.dt,
        "splits": {
            name: {
                **split,
                "shape": [split["samples"], split["horizon"] + 1, 1]
                + [spec.n] * spec.ndim,
            }
            for name, split in SPLITS.items()
        },
        "solver": SOLVER_POLICY,
        "qa_policy": QA_POLICY,
        "apebench_commit": APEBENCH_COMMIT,
        "versions": PINNED_VERSIONS,
        "exponax_semantics_source_sha256": EX_SOURCE_HASHES,
        "old_anchor_policy": "Untouched: this registry and new keys do not modify or regenerate old datasets.",
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _source_tree_hash(path: Path) -> str:
    return object_hash(
        {
            str(p.relative_to(path)).replace("\\", "/"): file_hash(p)
            for p in sorted(path.rglob("*.py"))
        }
    )


def load_runtime(apebench_repo: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Fail closed before solver use; read-only inspection never invokes a rollout."""
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    import jax.numpy as jnp
    import apebench
    import exponax as ex
    from apebench.scenarios import physical

    versions = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
    if versions != PINNED_VERSIONS:
        raise RuntimeError(f"Pinned environment mismatch: {versions}")
    if (
        jax.default_backend() != "cpu"
        or not jax.config.x64_enabled
        or jnp.asarray(0.0).dtype != jnp.float64
    ):
        raise RuntimeError(
            "Reference generation requires JAX CPU x64 with float64 defaults"
        )
    repo = apebench_repo.resolve()
    imported = Path(apebench.__file__).resolve()
    if not imported.is_relative_to(repo / "apebench"):
        raise RuntimeError(
            f"Imported APEBench is outside requested checkout: {imported}"
        )
    if _git(repo, "rev-parse", "HEAD") != APEBENCH_COMMIT:
        raise RuntimeError("APEBench commit mismatch")
    if _git(repo, "status", "--porcelain", "--untracked-files=no", "--", "apebench"):
        raise RuntimeError("Tracked APEBench source differs from pinned commit")
    exroot = Path(ex.__file__).resolve().parent
    audited_hashes = {
        relative: file_hash(exroot / relative) for relative in EX_SOURCE_HASHES
    }
    if audited_hashes != EX_SOURCE_HASHES:
        raise RuntimeError("Audited Exponax convection/diffusion source hash mismatch")
    provenance = {
        "versions": versions,
        "apebench_commit": APEBENCH_COMMIT,
        "apebench_repo": str(repo),
        "exponax_root": str(exroot),
        "apebench_source_tree_sha256": _source_tree_hash(repo / "apebench"),
        "exponax_source_tree_sha256": _source_tree_hash(exroot),
        "source_semantics_sha256": audited_hashes,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "backend": jax.default_backend(),
            "x64": bool(jax.config.x64_enabled),
        },
    }
    return (jax, ex, physical, provenance)


def make_scenario(
    task: str,
    physical: Any,
    *,
    n: int | None = None,
    substep_multiplier: int = 1,
    dealiasing: float | None = None,
) -> Any:
    spec = TASKS[task]
    kwargs = {
        "num_spatial_dims": spec.ndim,
        "num_points": spec.n if n is None else n,
        "domain_extent": spec.length,
        "dt": spec.dt,
        "num_channels": 1,
        "ic_config": "fourier;5;true;true",
        "num_warmup_steps": spec.warmup,
        "train_seed": 91001,
        "test_seed": 91003,
        "num_train_samples": 50,
        "num_test_samples": 30,
        "train_temporal_horizon": 50,
        "test_temporal_horizon": 1000,
    }
    if spec.matrix is not None:
        kwargs["diffusion_coef_matrix"] = spec.matrix
    else:
        kwargs.update(
            convection_coef=spec.convection,
            diffusion_coef=spec.diffusion,
            conservative=True,
            order=SOLVER_POLICY["nonlinear_order"],
            num_substeps=SOLVER_POLICY["nonlinear_substeps"] * substep_multiplier,
            dealiasing_fraction=SOLVER_POLICY["dealiasing_fraction"]
            if dealiasing is None
            else dealiasing,
            num_circle_points=SOLVER_POLICY["num_circle_points"],
            circle_radius=1.0,
        )
        if spec.family == "conservative_ks":
            kwargs["hyp_diffusion_coef"] = spec.hyperdiffusion
    scenario = getattr(physical, spec.scenario_class)(**kwargs)
    if scenario.num_channels != 1 or scenario.num_points != kwargs["num_points"]:
        raise RuntimeError("Effective scenario shape differs from contract")
    if spec.matrix is None:
        if (
            list(scenario.a_coefs)
            != effective_coefficients(spec)["linear_derivative_coefficients"]
        ):
            raise RuntimeError(
                "Effective linear coefficients differ from frozen physical task"
            )
    return scenario


def describe_solver(stepper: Any) -> dict[str, Any]:
    names = (
        "num_spatial_dims",
        "domain_extent",
        "num_points",
        "num_channels",
        "dt",
        "dx",
        "num_sub_steps",
        "linear_coefficients",
        "convection_scale",
        "diffusivity",
        "conservative",
        "single_channel",
        "dealiasing_fraction",
        "order",
        "num_circle_points",
        "circle_radius",
    )
    result = {
        "class": f"{type(stepper).__module__}.{type(stepper).__name__}",
        "parameters": {},
    }
    for name in names:
        if hasattr(stepper, name):
            value = getattr(stepper, name)
            if hasattr(value, "shape"):
                value = np.asarray(value).tolist()
            elif isinstance(value, tuple):
                value = list(value)
            result["parameters"][name] = value
    if hasattr(stepper, "stepper"):
        result["nested_stepper"] = describe_solver(stepper.stepper)
    return result


def assert_finite(name: str, data: np.ndarray) -> None:
    if not np.isfinite(data).all():
        raise RuntimeError(
            f"{name}: non-finite reference data; no automatic retry or adjustment"
        )


def _rollout(jax: Any, ex: Any, stepper: Any, initial: Any, horizon: int) -> np.ndarray:
    result = np.asarray(
        jax.jit(jax.vmap(ex.rollout(stepper, horizon, include_init=True)))(initial)
    )
    assert_finite("rollout", result)
    if result.dtype != np.float64:
        raise RuntimeError(f"Integration dtype must be float64, got {result.dtype}")
    return result


def _initial(
    jax: Any, ex: Any, scenario: Any, seed: int, samples: int, *, warmup: bool
) -> np.ndarray:
    initial = ex.build_ic_set(
        scenario.get_ic_generator(),
        num_points=scenario.num_points,
        num_samples=samples,
        key=jax.random.PRNGKey(seed),
    )
    if warmup and scenario.num_warmup_steps:
        initial = jax.jit(
            jax.vmap(ex.repeat(scenario.get_ref_stepper(), scenario.num_warmup_steps))
        )(initial)
    result = np.asarray(initial)
    assert_finite("initial/warmed state", result)
    if result.dtype != np.float64:
        raise RuntimeError("IC generator did not produce float64 with x64 enabled")
    return result


def prolong_periodic_1d(data: np.ndarray) -> np.ndarray:
    """Evaluate the same trigonometric polynomial on a 2N grid (Nyquist split)."""
    n = data.shape[-1]
    if n % 2:
        raise ValueError("Even grid required")
    spectrum = np.fft.rfft(data, axis=-1)
    fine = np.zeros(data.shape[:-1] + (n + 1,), dtype=np.complex128)
    fine[..., : n // 2 + 1] = 2 * spectrum
    fine[..., n // 2] *= 0.5
    return np.fft.irfft(fine, n=2 * n, axis=-1)


def reference_to_observation(task: str, data: np.ndarray) -> np.ndarray:
    spec, grid = (TASKS[task], grid_contract(task))
    if (
        list(data.shape[-spec.ndim :])
        != [grid["reference_points_per_axis"]] * spec.ndim
    ):
        raise ValueError("Reference-grid shape differs from frozen contract")
    slices = (slice(None),) * (data.ndim - spec.ndim) + (
        slice(None, None, grid["restriction_stride"]),
    ) * spec.ndim
    return data[slices]


def _reference_initial(
    task: str, jax: Any, ex: Any, physical: Any, seed: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep original IC keys/grid; lift before the one reference-grid warmup."""
    spec, grid = (TASKS[task], grid_contract(task))
    observed_scenario = make_scenario(task, physical)
    raw = _initial(jax, ex, observed_scenario, seed, samples, warmup=False)
    initial = raw
    while initial.shape[-1] < grid["reference_points_per_axis"]:
        initial = prolong_periodic_1d(initial)
    if spec.warmup:
        reference_scenario = make_scenario(
            task, physical, n=grid["reference_points_per_axis"]
        )
        initial = np.asarray(
            jax.jit(
                jax.vmap(ex.repeat(reference_scenario.get_ref_stepper(), spec.warmup))
            )(initial)
        )
    assert_finite("reference initial/warmed state", initial)
    if initial.dtype != np.float64:
        raise RuntimeError("Reference initial state must be float64")
    reference_to_observation(task, initial)
    return (initial, raw)


def relative_rms_error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Max over individual IC/time; never masks a bad IC with a global mean."""
    if candidate.shape != reference.shape:
        raise ValueError("Paired comparison requires equal shape")
    assert_finite("candidate", candidate)
    assert_finite("reference", reference)
    axes = tuple(range(2, reference.ndim))
    rmse = np.sqrt(np.mean((candidate - reference) ** 2, axis=axes))
    reference_rms = np.sqrt(np.mean(reference**2, axis=axes))
    relative = rmse / np.maximum(reference_rms, QA_POLICY["relative_rms_floor"])
    return {
        "max_relative_rms": float(relative.max()),
        "max_absolute_rms": float(rmse.max()),
        "relative_rms_per_ic_time": relative.tolist(),
        "reference_rms_per_ic_time": reference_rms.tolist(),
    }


def analytic_diffusion(initial: np.ndarray, spec: TaskSpec, horizon: int) -> np.ndarray:
    if spec.matrix is None:
        raise ValueError("Analytic diffusion requires a tensor task")
    k = 2 * np.pi * np.fft.fftfreq(spec.n, d=spec.length / spec.n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    matrix = np.asarray(spec.matrix)
    symbol = -(matrix[0, 0] * kx**2 + 2 * matrix[0, 1] * kx * ky + matrix[1, 1] * ky**2)
    spectrum = np.fft.fftn(initial, axes=(-2, -1))
    return np.stack(
        [
            np.fft.ifftn(spectrum * np.exp(i * spec.dt * symbol), axes=(-2, -1)).real
            for i in range(horizon + 1)
        ],
        axis=1,
    )


def _mass_drift(trajectory: np.ndarray) -> float:
    means = trajectory.mean(axis=tuple(range(2, trajectory.ndim)))
    return float(np.max(np.abs(means - means[:, :1])))


def qa_checks_pass(checks: list[dict[str, Any]]) -> bool:
    return bool(checks) and all(
        (np.isfinite(c["value"]) and c["value"] <= c["limit"] for c in checks)
    )


def run_qa(
    task: str, jax: Any, ex: Any, physical: Any, provenance: dict[str, Any]
) -> dict[str, Any]:
    spec = TASKS[task]
    grid = grid_contract(task)
    reference_n = grid["reference_points_per_axis"]
    scenario = make_scenario(task, physical, n=reference_n)
    report = {
        "schema": "sgno-regime-reference-qa-v1",
        "task_alias": task,
        "contract_sha256": object_hash(contract(task)),
        "generator_sha256": file_hash(Path(__file__)),
        "source": provenance,
        "policy": QA_POLICY,
        "grid": grid,
        "checks": [],
        "comparisons": {},
        "status": "FAIL",
        "scientific_scope": QA_POLICY["scope"],
    }

    def check(name: str, value: float, limit_key: str) -> None:
        report["checks"].append(
            {
                "name": name,
                "value": float(value),
                "limit": QA_POLICY[limit_key],
                "pass": bool(np.isfinite(value) and value <= QA_POLICY[limit_key]),
            }
        )

    def comparison(
        name: str, first: np.ndarray, second: np.ndarray, limit_key: str
    ) -> float:
        metrics = relative_rms_error(first[:, 1:], second[:, 1:])
        report["comparisons"][name] = metrics
        value = metrics["max_relative_rms"]
        check(name, value, limit_key)
        return value

    try:
        initial, raw_initial = _reference_initial(
            task, jax, ex, physical, QA_POLICY["seed"], QA_POLICY["samples"]
        )
        report["initial_sha256"] = array_hash(initial)
        report["raw_observation_grid_ic_sha256"] = array_hash(raw_initial)
        report["full_reference_initial_float64_sha256"] = array_hash(initial)
        report["observed_initial_float64_sha256"] = array_hash(
            reference_to_observation(task, initial)
        )
        coarse = _rollout(
            jax, ex, scenario.get_ref_stepper(), initial, QA_POLICY["horizon"]
        )
        check("mass_drift", _mass_drift(coarse), "mass_max_absolute_drift")
        observed = reference_to_observation(task, coarse)
        comparison(
            "float32_rounding",
            observed.astype(np.float32).astype(np.float64),
            observed,
            "float32_max_relative_rms",
        )
        if spec.matrix is not None:
            analytic = analytic_diffusion(initial, spec, QA_POLICY["horizon"])
            comparison(
                "analytic_diffusion", coarse, analytic, "analytic_max_relative_rms"
            )
            x = np.arange(spec.n) * spec.length / spec.n
            xx, yy = np.meshgrid(x, x, indexing="ij")
            probe = (
                0.37
                + np.sin(2 * np.pi * (xx + 2 * yy) / spec.length)
                + 0.3 * np.cos(2 * np.pi * (2 * xx - yy) / spec.length)
            )[None, None]
            probe_num = _rollout(
                jax, ex, scenario.get_ref_stepper(), probe, QA_POLICY["horizon"]
            )
            comparison(
                "analytic_mixed_mode_and_zero_mode",
                probe_num,
                analytic_diffusion(probe, spec, QA_POLICY["horizon"]),
                "analytic_max_relative_rms",
            )
        else:
            fine_stepper = make_scenario(
                task, physical, n=reference_n, substep_multiplier=2
            ).get_ref_stepper()
            finer_stepper = make_scenario(
                task, physical, n=reference_n, substep_multiplier=4
            ).get_ref_stepper()
            fine = _rollout(jax, ex, fine_stepper, initial, QA_POLICY["horizon"])
            finer = _rollout(jax, ex, finer_stepper, initial, QA_POLICY["horizon"])
            e1 = comparison(
                "time_10_vs_20_substeps", coarse, fine, "time_coarse_max_relative_rms"
            )
            e2 = comparison(
                "time_20_vs_40_substeps", fine, finer, "time_fine_max_relative_rms"
            )
            ratio = (
                0.0
                if e1 < QA_POLICY["time_convergence_ratio_noise_floor"]
                and e2 < QA_POLICY["time_convergence_ratio_noise_floor"]
                else e2 / max(e1, 1e-30)
            )
            check("time_error_reduction_ratio", ratio, "time_convergence_ratio_max")
            initial_high = prolong_periodic_1d(initial)
            high_stepper = make_scenario(
                task,
                physical,
                n=grid["qa_refined_points_per_axis"],
                substep_multiplier=4,
            ).get_ref_stepper()
            high = _rollout(jax, ex, high_stepper, initial_high, QA_POLICY["horizon"])
            comparison(
                "reference_grid_vs_refined_grid_common_points",
                finer,
                high[..., ::2],
                "grid_max_relative_rms",
            )
            strict_stepper = make_scenario(
                task, physical, n=reference_n, substep_multiplier=4, dealiasing=0.5
            ).get_ref_stepper()
            strict = _rollout(jax, ex, strict_stepper, initial, QA_POLICY["horizon"])
            comparison(
                "dealias_2thirds_vs_half", finer, strict, "dealias_max_relative_rms"
            )
            cutoff = SOLVER_POLICY["dealiasing_fraction"] * (reference_n // 2) - 1
            high_hat = np.fft.fft(high, axis=-1)
            modes = (
                np.fft.fftfreq(grid["qa_refined_points_per_axis"])
                * grid["qa_refined_points_per_axis"]
            )
            energy = np.abs(high_hat) ** 2
            tail = energy[..., np.abs(modes) > cutoff].sum(axis=(-2, -1)) / np.maximum(
                energy.sum(axis=(-2, -1)), 1e-30
            )
            report["high_grid_tail_energy_per_ic_time"] = tail.tolist()
            check(
                "high_grid_unresolved_energy",
                float(tail.max()),
                "high_grid_unresolved_energy_fraction",
            )
        report["status"] = "PASS" if qa_checks_pass(report["checks"]) else "FAIL"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def validate_qa_report(
    task: str, report: dict[str, Any], provenance: dict[str, Any]
) -> None:
    if report.get("status") != "PASS" or not qa_checks_pass(report.get("checks", [])):
        raise RuntimeError("Reference QA is not PASS; generation is blocked")
    if (
        report.get("contract_sha256") != object_hash(contract(task))
        or report.get("task_alias") != task
    ):
        raise RuntimeError("Reference QA contract/task mismatch")
    if report.get("generator_sha256") != file_hash(Path(__file__)):
        raise RuntimeError("Generator changed after reference QA")
    expected = {
        "mass_drift": "mass_max_absolute_drift",
        "float32_rounding": "float32_max_relative_rms",
    }
    if TASKS[task].matrix is not None:
        expected.update(
            analytic_diffusion="analytic_max_relative_rms",
            analytic_mixed_mode_and_zero_mode="analytic_max_relative_rms",
        )
    else:
        expected.update(
            time_10_vs_20_substeps="time_coarse_max_relative_rms",
            time_20_vs_40_substeps="time_fine_max_relative_rms",
            time_error_reduction_ratio="time_convergence_ratio_max",
            reference_grid_vs_refined_grid_common_points="grid_max_relative_rms",
            dealias_2thirds_vs_half="dealias_max_relative_rms",
            high_grid_unresolved_energy="high_grid_unresolved_energy_fraction",
        )
    checks = report["checks"]
    if len(checks) != len(expected) or {c.get("name") for c in checks} != set(expected):
        raise RuntimeError("Reference QA is missing required checks")
    for check in checks:
        if (
            check.get("limit") != QA_POLICY[expected[check["name"]]]
            or check.get("pass") is not True
        ):
            raise RuntimeError(
                "Reference QA check threshold or pass flag differs from policy"
            )
    if report.get("policy") != QA_POLICY:
        raise RuntimeError("Reference QA policy mismatch")
    if report.get("grid") != grid_contract(task):
        raise RuntimeError("Reference QA grid policy mismatch")
    for key in (
        "raw_observation_grid_ic_sha256",
        "full_reference_initial_float64_sha256",
        "observed_initial_float64_sha256",
    ):
        if not isinstance(report.get(key), str) or len(report[key]) != 64:
            raise RuntimeError(f"Reference QA missing initial-state provenance: {key}")
    for key in (
        "versions",
        "apebench_commit",
        "apebench_source_tree_sha256",
        "exponax_source_tree_sha256",
        "source_semantics_sha256",
    ):
        if report.get("source", {}).get(key) != provenance.get(key):
            raise RuntimeError(f"Reference QA source mismatch: {key}")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def require_fresh_output(path: Path) -> None:
    if path.suffix != ".npz":
        raise ValueError("Output must have .npz suffix")
    candidates = [
        path,
        path.with_suffix(".metadata.json"),
        path.with_suffix(".metadata.sha256"),
    ]
    if any((p.exists() for p in candidates)):
        raise FileExistsError(
            "Frozen output or sidecar already exists; refusing overwrite"
        )


def generate(
    task: str,
    out: Path,
    qa_path: Path,
    expected_hash: str,
    jax: Any,
    ex: Any,
    physical: Any,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    require_fresh_output(out)
    if expected_hash != object_hash(contract(task)):
        raise RuntimeError("Explicit frozen contract hash mismatch")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    validate_qa_report(task, qa, provenance)
    grid = grid_contract(task)
    scenario = make_scenario(task, physical, n=grid["reference_points_per_axis"])
    stepper = scenario.get_ref_stepper()
    arrays, split_stats = ({}, {})
    for name, split in SPLITS.items():
        initial, raw_initial = _reference_initial(
            task, jax, ex, physical, split["seed"], split["samples"]
        )
        observed_initial = reference_to_observation(task, initial)
        shape = tuple(contract(task)["splits"][name]["shape"])
        target = np.empty(shape, dtype=np.float32)
        mass_max, round_max = (0.0, 0.0)
        for start in range(0, split["samples"], 2):
            trajectory = _rollout(
                jax, ex, stepper, initial[start : start + 2], split["horizon"]
            )
            observed = reference_to_observation(task, trajectory)
            stored = observed.astype(np.float32)
            assert_finite(name, stored)
            target[start : start + 2] = stored
            mass_max = max(mass_max, _mass_drift(trajectory))
            rms0 = np.maximum(
                np.sqrt(
                    np.mean(
                        observed[:, 0] ** 2, axis=tuple(range(1, observed.ndim - 1))
                    )
                ),
                1e-12,
            )
            rounding = np.sqrt(
                np.mean(
                    (stored.astype(np.float64) - observed) ** 2,
                    axis=tuple(range(2, observed.ndim)),
                )
            )
            round_max = max(round_max, float(np.max(rounding / rms0[:, None])))
        if mass_max > QA_POLICY["mass_max_absolute_drift"]:
            raise RuntimeError(
                f"{name}: long-horizon mass drift {mass_max} failed; output not frozen"
            )
        if round_max > QA_POLICY["float32_max_relative_rms"]:
            raise RuntimeError(f"{name}: storage rounding {round_max} failed")
        arrays[name] = target
        split_stats[name] = {
            "shape": list(target.shape),
            "dtype": str(target.dtype),
            "sha256": array_hash(target),
            "initial_float64_sha256": array_hash(observed_initial),
            "initial_saved_sha256": array_hash(target[:, 0]),
            "raw_observation_grid_ic_sha256": array_hash(raw_initial),
            "full_reference_initial_float64_sha256": array_hash(initial),
            "observed_initial_float64_sha256": array_hash(observed_initial),
            "full_reference_initial_shape": list(initial.shape),
            "observed_initial_shape": list(observed_initial.shape),
            "max_absolute": float(np.max(np.abs(target))),
            "all_finite": True,
            "max_mass_drift_float64": mass_max,
            "mass_drift_grid": "full_reference_grid",
            "max_rounding_rms_relative_to_initial": round_max,
            "rounding_grid": "observation_grid",
        }
        print(json.dumps({"split_completed": name, **split_stats[name]}), flush=True)
    metadata = {
        "schema": "sgno-regime-frozen-data-v1",
        "status": "COMPLETE",
        "contract": contract(task),
        "contract_sha256": expected_hash,
        "generator_sha256": file_hash(Path(__file__)),
        "source": provenance,
        "reference_qa_path": str(qa_path.resolve()),
        "reference_qa_sha256": file_hash(qa_path),
        "reference_qa_status": qa["status"],
        "splits": split_stats,
        "grid": grid,
        "reference_solver": describe_solver(stepper),
        "scenario_class": f"{type(scenario).__module__}.{type(scenario).__name__}",
        "explicit_registry_name": task,
        "upstream_scenario_name": scenario.get_scenario_name(),
        "upstream_name_note": "Pinned Burgers.get_scenario_name lacks return; explicit registry alias is authoritative, upstream is not patched.",
        "split_access_policy": "train only for updates; validation retained but not used for optimizer/checkpoint/configuration selection; test only frozen final-checkpoint evaluation, never configuration selection.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["npz_sha256"] = file_hash(out)
    metadata_path = out.with_suffix(".metadata.json")
    write_json_exclusive(metadata_path, metadata)
    with out.with_suffix(".metadata.sha256").open("x", encoding="ascii") as stream:
        stream.write(file_hash(metadata_path) + "\n")
    verify_artifact(out)
    return metadata


def verify_artifact(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(".metadata.json")
    expected_meta_hash = (
        path.with_suffix(".metadata.sha256").read_text(encoding="ascii").strip()
    )
    if file_hash(metadata_path) != expected_meta_hash:
        raise RuntimeError("Metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("status") != "COMPLETE"
        or file_hash(path) != metadata["npz_sha256"]
    ):
        raise RuntimeError("Incomplete artifact or NPZ hash mismatch")
    task = metadata["contract"]["task_alias"]
    if metadata["contract_sha256"] != object_hash(contract(task)) or metadata[
        "contract_sha256"
    ] != object_hash(metadata["contract"]):
        raise RuntimeError("Artifact has another data contract")
    if metadata.get("grid") != grid_contract(task):
        raise RuntimeError("Artifact reference/observation grid metadata mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(SPLITS):
            raise RuntimeError(
                "Dataset split keys differ from train/validation/test contract"
            )
        for name in SPLITS:
            data = archive[name]
            if (
                list(data.shape) != contract(task)["splits"][name]["shape"]
                or data.dtype != np.float32
            ):
                raise RuntimeError(f"{name}: shape/dtype mismatch")
            assert_finite(name, data)
            if array_hash(data) != metadata["splits"][name]["sha256"]:
                raise RuntimeError(f"{name}: array hash mismatch")
            info = metadata["splits"][name]
            if array_hash(data[:, 0]) != info["initial_saved_sha256"]:
                raise RuntimeError(f"{name}: saved initial-state hash mismatch")
            if (
                info["initial_float64_sha256"]
                != info["observed_initial_float64_sha256"]
            ):
                raise RuntimeError(f"{name}: observed initial-state hash mismatch")
            spec, grid = (TASKS[task], grid_contract(task))
            if (
                info["full_reference_initial_shape"]
                != [SPLITS[name]["samples"], 1]
                + [grid["reference_points_per_axis"]] * spec.ndim
            ):
                raise RuntimeError(f"{name}: full-reference initial shape mismatch")
            if info["observed_initial_shape"] != list(data[:, 0].shape):
                raise RuntimeError(f"{name}: observation initial shape mismatch")
    return {"status": "VERIFIED", "task": task, "npz_sha256": metadata["npz_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("describe", "audit", "qa", "generate", "verify"),
        required=True,
    )
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--apebench-repo", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--qa-report", type=Path)
    parser.add_argument("--expected-contract-sha256")
    args = parser.parse_args()
    if args.action == "verify":
        if args.out is None:
            parser.error("verify requires --out existing.npz")
        print(json.dumps(verify_artifact(args.out), indent=2))
        return
    if not args.task:
        parser.error("--task is required")
    if args.action == "describe":
        result = {
            "contract": contract(args.task),
            "contract_sha256": object_hash(contract(args.task)),
            "generator_sha256": file_hash(Path(__file__)),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.apebench_repo is None:
        parser.error("--apebench-repo is required for runtime source verification")
    if args.action in ("qa", "generate") and args.out is None:
        parser.error("--out is required")
    if args.action == "generate" and (
        args.qa_report is None or not args.expected_contract_sha256
    ):
        parser.error("generate requires --qa-report and --expected-contract-sha256")
    if args.out is not None and args.out.exists() and (args.action != "audit"):
        parser.error("Output already exists; no overwriting or automatic retries")
    jax, ex, physical, provenance = load_runtime(args.apebench_repo)
    if args.action == "audit":
        scenario = make_scenario(args.task, physical)
        print(
            json.dumps(
                {
                    "status": "SOURCE_AUDITED",
                    "source": provenance,
                    "contract_sha256": object_hash(contract(args.task)),
                    "upstream_name": scenario.get_scenario_name(),
                },
                indent=2,
            )
        )
    elif args.action == "qa":
        result = run_qa(args.task, jax, ex, physical, provenance)
        write_json_exclusive(args.out, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "checks": result["checks"],
                    "error": result.get("error"),
                    "report": str(args.out),
                },
                indent=2,
            )
        )
        if result["status"] != "PASS":
            raise SystemExit(2)
    else:
        result = generate(
            args.task,
            args.out,
            args.qa_report,
            args.expected_contract_sha256,
            jax,
            ex,
            physical,
            provenance,
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "npz_sha256": result["npz_sha256"],
                    "out": str(args.out),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
