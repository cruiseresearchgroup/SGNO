import torch
from sgno import build_sgno_from_config, summarize_gain_audit

def _run_1d():
    cfg = "sgno_canonical;width=8;modes=8;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 1, 128, 1)
    u = torch.randn(2, 1, 128)
    y = m(u)
    assert y.shape == u.shape
    audit = summarize_gain_audit(m)
    assert "gamma_min" in audit

def _run_2d():
    cfg = "sgno_canonical;width=8;modes=4;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 2, 32, 1)
    u = torch.randn(2, 1, 32, 32)
    y = m(u)
    assert y.shape == u.shape

def _run_3d():
    cfg = "sgno_canonical;width=4;modes=4;n_blocks=1;initial_step=1"
    m = build_sgno_from_config(cfg, 3, 16, 1)
    u = torch.randn(1, 1, 16, 16, 16)
    y = m(u)
    assert y.shape == u.shape

def main():
    torch.set_grad_enabled(False)
    _run_1d()
    _run_2d()
    _run_3d()
    print("ok")

if __name__ == "__main__":
    main()
