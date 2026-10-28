import importlib.util
import json
import sys
import csv
from pathlib import Path

import numpy as np
import pytest

from sgno.metrics import field_diagnostics


@pytest.mark.parametrize("shape", [(2, 1, 15), (2, 1, 16), (2, 2, 12, 14), (2, 1, 6, 7, 8)])
def test_spectral_and_field_identities(shape):
    rng = np.random.default_rng(25)
    a, b = rng.normal(size=shape), rng.normal(size=shape)
    d = field_diagnostics(a, b)
    axes = tuple(range(1, len(shape)))
    expected = np.sum((a-b)**2, axis=axes) / np.sum(b**2, axis=axes)
    np.testing.assert_allclose(d["total_squared"], expected, atol=1e-12)
    np.testing.assert_allclose(d["amplitude_squared"] + d["phase_squared"], expected, atol=1e-12)
    np.testing.assert_allclose(1 + d["squared_norm_ratio"] - 2*np.sqrt(d["squared_norm_ratio"])*d["field_correlation"], expected, atol=1e-12)


def test_translation_and_amplitude_have_distinct_components():
    x = np.arange(32) / 32
    reference = np.cos(2*np.pi*x)[None, None]
    shifted = np.cos(2*np.pi*(x - 0.125))[None, None]
    phase = field_diagnostics(shifted, reference)
    assert phase["amplitude_squared"][0] < 1e-25
    assert phase["phase_squared"][0] == pytest.approx(2-2*np.cos(np.pi/4))
    amplitude = field_diagnostics(2*reference, reference)
    assert amplitude["amplitude_squared"][0] == pytest.approx(1)
    assert amplitude["phase_squared"][0] == pytest.approx(0, abs=1e-14)
    assert amplitude["squared_norm_ratio"][0] == pytest.approx(4)
    assert amplitude["field_correlation"][0] == pytest.approx(1)


def test_nonfinite_and_undefined_diagnostics_are_retained():
    reference = np.ones((2, 1, 8))
    prediction = reference.copy()
    prediction[1] = np.nan
    d = field_diagnostics(prediction, reference)
    assert np.isnan(d["total_squared"][1])
    d = field_diagnostics(np.zeros_like(reference), reference)
    np.testing.assert_allclose(d["total_squared"], 1)
    assert np.isnan(d["field_correlation"]).all()


def test_aggregate_median_spread_pairing_and_nonfinite(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/aggregate.py"
    spec = importlib.util.spec_from_file_location("aggregate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = [3, 1, 100, 2, 4]
    for seed in reversed(range(5)):
        run = tmp_path / f"seed{seed}"
        run.mkdir()
        data = dict(task="example", corrector="sgno", outer="fitted_joint", seed=seed,
                    steps=200, gmean_1_100=values[seed], endpoint_nrmse="inf" if seed==2 else 1)
        (run / "summary.json").write_text(json.dumps(data))
    out = tmp_path / "aggregated.csv"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--runs", str(tmp_path), "--out", str(out)])
    module.main()
    rows = list(csv.DictReader(out.open()))
    finite = next(r for r in rows if r["metric"] == "gmean_1_100")
    assert float(finite["median"]) == 3
    assert float(finite["mean"]) == 22
    assert float(finite["sample_sd"]) == pytest.approx(np.std(values, ddof=1))
    assert float(finite["run_A"]) == 3
    nonfinite = next(r for r in rows if r["metric"] == "endpoint_nrmse")
    assert nonfinite["median"] == nonfinite["mean"] == "NF"
    assert nonfinite["run_C"] == "NF"
    assert nonfinite["nonfinite_seeds"] == "1"
