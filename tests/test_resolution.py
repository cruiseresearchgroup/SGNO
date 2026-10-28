import copy
import numpy as np
import pytest
import torch
from sgno import build_model
from sgno.resolution import FrozenGridPredictor, resize


def field(n):
    x = torch.arange(n) / n
    return (torch.sin(2*torch.pi*x)[:, None] + 0.2*torch.cos(6*torch.pi*x)[None, :])[None, None]


@pytest.mark.parametrize("source,target", [(32,128),(128,32),(64,32),(64,128)])
def test_fourier_resampling_preserves_low_modes(source, target):
    torch.testing.assert_close(resize(field(source), target), field(target), atol=2e-6, rtol=2e-6)


def test_nyquist_split_and_merge():
    x = ((-1.)**torch.arange(32))[None, None, :, None].expand(1, 1, 32, 32)
    torch.testing.assert_close(resize(resize(x, 64), 32), x, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("grid,policy", [(64,"frequency_direct"),(32,"frequency_direct"),(128,"frequency_direct"),(32,"project_to64"),(128,"project_to64")])
def test_transfer_freezes_parameters_and_preserves_native_result(grid, policy):
    torch.set_num_threads(1)
    model = build_model(2, 0)
    data = np.random.default_rng(8).normal(size=(2, 3, 1, 64, 64)).astype(np.float32)
    model.initialize_from_train(data)
    with torch.no_grad():
        model.residual.predictor.base.core.fc2.weight.fill_(1e-3)
        model.eval()
        expected = model(field(64))
    before = {k:p.clone() for k,p in model.named_parameters()}
    adapter = FrozenGridPredictor(model, grid, policy)
    result = adapter(field(grid))
    assert result.shape == (1,1,grid,grid)
    assert torch.isfinite(result).all()
    for name, parameter in model.named_parameters():
        assert not parameter.requires_grad
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    if grid == 64:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        adapter.train()
