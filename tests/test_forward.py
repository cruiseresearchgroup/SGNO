import pytest
import torch
from sgno import build_sgno_from_config, summarize_gain_audit
from sgno.sgno import _phi1, _project_rfft_hermitian_boundaries

def test_forward_1d():
    cfg = "sgno_canonical;width=8;modes=8;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 1, 64, 1)
    u = torch.randn(2, 1, 64)
    y = m(u)
    assert y.shape == u.shape

def test_forward_2d():
    cfg = "sgno_canonical;width=8;modes=4;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 2, 16, 1)
    u = torch.randn(2, 1, 16, 16)
    y = m(u)
    assert y.shape == u.shape

def test_gain_audit():
    cfg = "sgno_canonical;width=4;modes=4;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 1, 32, 1)
    audit = summarize_gain_audit(m)
    assert "gamma_min" in audit
    assert "mu_max" in audit
    assert audit["certifies_full_map"] is False
    assert audit["q_delta_heuristic"] == audit["q_delta"]
    assert "not the paper's full-map Lipschitz bound" in audit["scope_warning"]


def test_public_nd_anchor_rejects_overlapping_or_clipped_mode_blocks():
    model_2d = build_sgno_from_config(
        "sgno_canonical;width=2;modes=5;n_blocks=1;initial_step=1", 2, 8, 1
    )
    with pytest.raises(ValueError, match="overlap"):
        model_2d(torch.randn(1, 1, 8, 8))

    model_3d = build_sgno_from_config(
        "sgno_canonical;width=2;modes=4;n_blocks=1;initial_step=1", 3, 6, 1
    )
    with pytest.raises(ValueError, match="overlap"):
        model_3d(torch.randn(1, 1, 6, 6, 6))


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_phi1_has_correct_value_and_finite_gradient_at_zero(dtype):
    z = torch.zeros(1, dtype=dtype, requires_grad=True)
    value = _phi1(z)
    assert torch.allclose(value, torch.ones_like(value))
    gradient = torch.autograd.grad(value.real.sum(), z)[0]
    assert torch.isfinite(gradient).all()
    assert torch.allclose(gradient.real, torch.full_like(gradient.real, 0.5))


@pytest.mark.parametrize("shape", [(7,), (5, 6), (4, 5, 6), (5, 4, 7)])
def test_explicit_hermitian_projection_matches_implicit_irfft(shape):
    stored_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    raw = torch.randn(*stored_shape, dtype=torch.complex128, requires_grad=True)
    projected = _project_rfft_hermitian_boundaries(raw, shape)

    implicit = torch.fft.irfftn(raw, s=shape)
    explicit = torch.fft.irfftn(projected, s=shape)
    assert torch.allclose(explicit, implicit, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(
        torch.fft.rfftn(explicit, s=shape),
        projected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    implicit_gradient = torch.autograd.grad(implicit.square().sum(), raw, retain_graph=True)[0]
    explicit_gradient = torch.autograd.grad(explicit.square().sum(), raw)[0]
    assert torch.allclose(
        explicit_gradient,
        implicit_gradient,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_spectral_blocks_reject_zero_modes_and_mismatched_forcing_shape():
    from sgno import SpectralETD1d, SpectralETD2d, SpectralETD3d

    with pytest.raises(ValueError, match="positive"):
        SpectralETD1d(2, 0)
    with pytest.raises(ValueError, match="positive"):
        SpectralETD2d(2, 2, 0)
    with pytest.raises(ValueError, match="positive"):
        SpectralETD3d(2, 2, 2, 0)

    block = SpectralETD1d(2, 2)
    with pytest.raises(ValueError, match="same shape"):
        block(torch.randn(1, 2, 8), torch.randn(1, 2, 7))


@pytest.mark.parametrize(
    ("spatial_dim", "width", "modes", "blocks", "num_points", "expected"),
    [
        (1, 11, 26, 7, 2048, 30_573),
        (2, 5, 10, 9, 64, 56_402),
        (3, 10, 6, 2, 32, 192_827),
    ],
)
def test_paper_parameter_element_counts_remain_unchanged(
    spatial_dim, width, modes, blocks, num_points, expected
):
    model = build_sgno_from_config(
        f"sgno_canonical;width={width};modes={modes};n_blocks={blocks};initial_step=1",
        spatial_dim,
        num_points,
        1,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
