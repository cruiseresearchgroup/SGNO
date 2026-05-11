import torch
from sgno import build_sgno_from_config, summarize_gain_audit

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
