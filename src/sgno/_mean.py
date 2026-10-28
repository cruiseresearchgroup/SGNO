"""Mean-preserving prediction on centered physical fields."""

import torch
from sgno._affine import (
    LinearResidualSGNO,
    model_config as parent_config,
    build_residual,
    SPATIAL_SIZES,
)

METHOD = "sgno_mean_preserving_v1"
VERSION = "mean-preserving-affine-fourier-sgno-fluctuation-update-v1"


def model_config(ndim):
    c = parent_config(ndim)
    c["physical_propagation"]["version"] = VERSION
    c["physical_propagation"]["formula"] = "P0(u)+Q[L(Q(u))+c+alpha*N(Q(u))]"
    c["mean_preservation"] = dict(
        formula="P0(u)+Q[L(Q(u))+c+alpha*(F_hf(Q(u))-Q(u))], Q=I-P0",
        applied_during_training_and_inference=True,
        added_parameters=0,
        target_mean="Input spatial mean, independently per batch and channel",
        input_mean_not_forced_zero=True,
        mean_conservation_imposed=True,
        justification="All14 observed training trajectories preserve their nearly zero spatial means to numerical precision",
        physical_coefficients_supplied=False,
        task_switches=False,
        residual_weight_initial=0.0,
        training_only_affine_initialization_unchanged=True,
    )
    return c


class MeanPreservingSGNO(LinearResidualSGNO):
    def forward(self, u):
        axes = tuple(range(2, u.ndim))
        mean = u.mean(axes, keepdim=True)
        centered = u - mean
        prediction = self.linear(centered) + self.residual_weight * (
            self.residual(centered) - centered
        )
        return prediction - prediction.mean(axes, keepdim=True) + mean

    def learned_state_propagation(self):
        r = super().learned_state_propagation()
        r.update(
            scope="Mean-preserving prediction with learned affine Fourier and SGNO fluctuation dynamics",
            input_spatial_mean_preserved=True,
            constant_shift_gain=1.0,
            nonlinear_input_centered=True,
            output_mean_restored_from_input=True,
        )
        return r


def build_model(ndim, seed=0):
    if ndim not in SPATIAL_SIZES or seed != 0:
        raise ValueError("One fixed all14 seed0 candidate")
    return MeanPreservingSGNO(build_residual(ndim, seed), ndim)


def model_spec(model):
    c = model_config(model.ndim)
    elements = sum((p.numel() for p in model.parameters()))
    real = sum((p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
    assert (elements, real) == (c["tensor_entries"], c["real_parameters"])
    return dict(
        method=METHOD,
        version=VERSION,
        ndim=model.ndim,
        config=c,
        parameter_counts=dict(tensor_entries=elements, real_parameters=real),
        inherited_initialization="Exact parent parameters, RNG, alpha=0 and training-only affine fit",
        prediction="Input mean plus a zero-mean affine Fourier and SGNO update of the centered field",
    )
