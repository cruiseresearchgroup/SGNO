"""Propagation controls used in the paper's ablation table."""

import torch
from .model import build_model
from .baselines.fno_corrector import AffineFNOShell, _build_fno


class FixedIdentity:
    def linear(self, u):
        return u

    @torch.no_grad()
    def initialize_from_train(self, data):
        receipt = super().initialize_from_train(data)
        self.generator.zero_()
        self.forcing.zero_()
        self.generator.requires_grad_(False)
        self.forcing.requires_grad_(False)
        return receipt


class DirectDisk:
    def multiplier(self):
        return self.generator

    @torch.no_grad()
    def initialize_from_train(self, data):
        receipt = super().initialize_from_train(data)
        value = torch.exp(
            torch.complex(-self.generator.real.square(), self.generator.imag)
        )
        self.generator.copy_(value)
        if self.initial_kind == "identity":
            self.generator.fill_(1)
            self.forcing.zero_()
        return receipt

    @torch.no_grad()
    def project_constraints(self):
        if not self.generator.requires_grad:
            return
        self.generator.div_(self.generator.abs().clamp_min(1))


def build_predictor(ndim, seed=0, corrector="sgno", outer="fitted_joint"):
    """Build the full model or a controlled outer-map variant.

    `fitted_joint` is the paper's original radial-square parameterization.
    `direct_identity` and `direct_fitted` share a direct complex parameterization.
    `direct_fitted_frozen` freezes both fitted modal parameters.
    """
    if corrector not in ("sgno", "fno"):
        raise ValueError("corrector must be sgno or fno")
    torch.manual_seed(seed)
    model = (
        build_model(ndim, seed)
        if corrector == "sgno"
        else AffineFNOShell(_build_fno(ndim), ndim)
    )
    if outer == "identity_fixed":
        model.__class__ = type(
            "Identity" + type(model).__name__, (FixedIdentity, type(model)), {}
        )
    elif outer in ("direct_identity", "direct_fitted", "direct_fitted_frozen"):
        model.__class__ = type(
            "Direct" + type(model).__name__, (DirectDisk, type(model)), {}
        )
        model.initial_kind = "identity" if outer == "direct_identity" else "fitted"
        if outer == "direct_fitted_frozen":
            model.generator.requires_grad_(False)
            model.forcing.requires_grad_(False)
    elif outer != "fitted_joint":
        raise ValueError(f"Unknown outer map: {outer}")
    return model
