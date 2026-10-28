import numpy as np
import pytest
import torch
from sgno import build_model
from sgno.ablations import build_predictor
from sgno.components import apply
from sgno.metrics import relative_rms, summarize

torch.set_num_threads(1)


def training_array(ndim):
    n = {1: 160, 2: 64, 3: 32}[ndim]
    data = (
        np.random.default_rng(9).normal(size=(2, 3, 1) + (n,) * ndim).astype(np.float32)
    )
    data -= data.mean(tuple(range(3, data.ndim)), keepdims=True)
    return data


@pytest.mark.parametrize("ndim,count", [(1, 30683), (2, 58708), (3, 210966)])
def test_fitted_initial_prediction_and_mean(ndim, count):
    model = build_model(ndim, 0)
    assert sum(p.numel() for p in model.parameters()) == count
    data = training_array(ndim)
    model.initialize_from_train(data)
    x = torch.from_numpy(data[:, 0]) + 0.4
    axes = tuple(range(2, x.ndim))
    with torch.no_grad():
        model.eval()
        expected = model.linear(x - x.mean(axes, keepdim=True))
        expected = (
            expected - expected.mean(axes, keepdim=True) + x.mean(axes, keepdim=True)
        )
        actual = model(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.mean(axes), x.mean(axes), atol=1e-6, rtol=1e-6)


def test_direct_identity_can_learn_attenuation():
    model = build_predictor(1, 0, outer="direct_identity")
    data = training_array(1)
    model.initialize_from_train(data)
    x = torch.from_numpy(data[:, 0])
    pred = model(x)
    torch.testing.assert_close(pred, x, atol=2e-6, rtol=2e-6)
    (pred - 0.9 * x).square().mean().backward()
    assert model.generator.grad.real.abs().max() > 0


@pytest.mark.parametrize(
    "component", ["no_nonlinear_spectral_update", "no_local_pointwise_path"]
)
def test_component_intervention_preserves_state_shape(component):
    model = build_model(1, 0)
    data = training_array(1)
    model.initialize_from_train(data)
    receipt = apply(model, component)
    assert (
        receipt["trainable_parameter_elements"] < receipt["stored_parameter_elements"]
    )
    assert model(torch.from_numpy(data[:, 0])).shape == (2, 1, 160)


def test_fixed_map_and_matched_fno():
    model = build_predictor(1, 0, "fno", "identity_fixed")
    data = training_array(1)
    model.initialize_from_train(data)
    x = torch.from_numpy(data[:, 0])
    torch.testing.assert_close(model(x), x, atol=2e-6, rtol=2e-6)
    assert not model.generator.requires_grad and not model.forcing.requires_grad


def test_metrics_keep_nonfinite_outcomes():
    x = np.ones((2, 1, 8), np.float32)
    np.testing.assert_allclose(relative_rms(x * 1.1, x, 1), 0.1, atol=1e-7)
    errors = np.full((2, 200), 0.1)
    errors[1, 199] = np.inf
    result = summarize(errors)
    assert result["gmean_1_100"] == pytest.approx(0.1)
    assert np.isinf(result["gmean_101_200"])
    assert result["nonfinite_trajectory_steps"] == 1


def test_fitted_frozen_map_is_unchanged_after_optimizer_step():
    model = build_predictor(1, 0, "sgno", "direct_fitted_frozen")
    data = training_array(1)
    model.initialize_from_train(data)
    multiplier, intercept = model.generator.detach().clone(), model.forcing.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model(torch.from_numpy(data[:, 0])).square().mean().backward()
    optimizer.step()
    model.project_constraints()
    torch.testing.assert_close(model.generator, multiplier, atol=0, rtol=0)
    torch.testing.assert_close(model.forcing, intercept, atol=0, rtol=0)
