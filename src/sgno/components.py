"""Component removals of frozen zero-head SGNO; no changes to outer computation."""

import types
import torch

VARIANTS = ("no_nonlinear_spectral_update", "no_local_pointwise_path")


def _zero(self, x):
    return torch.zeros_like(x)


def apply(model, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if hasattr(model, "internal_ablation"):
        raise ValueError("Intervention already installed")
    from sgno._increment import ZeroHeadIncrementSGNO

    if not isinstance(model, ZeroHeadIncrementSGNO):
        raise TypeError("Requires the current frozen zero-head version")
    core = model.residual.predictor.base.core
    removed = []
    if variant == "no_nonlinear_spectral_update":
        for li, (mlp, spectral) in enumerate(zip(core.gs, core.etds)):
            mlp.forward = types.MethodType(_zero, mlp)
            for name, p in mlp.named_parameters():
                p.requires_grad_(False)
                removed.append(f"core.gs.{li}.{name}")
            for name, p in spectral.named_parameters():
                if name.startswith("mix") or name in ("inj_bias", "inj_slope_raw"):
                    p.requires_grad_(False)
                    removed.append(f"core.etds.{li}.{name}")
    else:
        for li, branch in enumerate(core.bs):
            branch.forward = types.MethodType(_zero, branch)
            for name, p in branch.named_parameters():
                p.requires_grad_(False)
                removed.append(f"core.bs.{li}.{name}")
    model.internal_ablation = variant
    return dict(
        variant=variant,
        intervened_layers=core.n_blocks,
        disabled_parameters=removed,
        stored_parameter_elements=sum(p.numel() for p in model.parameters()),
        trainable_parameter_elements=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        unchanged_outer_construction=True,
        unchanged_outer_initialization=True,
        surviving_parameters_jointly_trained=True,
        parameter_values_changed=False,
    )
