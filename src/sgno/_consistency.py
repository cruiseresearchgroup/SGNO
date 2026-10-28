"""Remove the high-frequency zero-input response of the correction."""

import torch
from torch import nn
from sgno._filter import build_model as build_adopted
from sgno._filter import model_config as adopted_config
from sgno._spec import ARCHITECTURES, SPATIAL_SIZES

METHOD = "sgno_highfreq_consistency_v1"
VERSION = "fixed-existing-modes-zero-response-highpass-subtraction-v1"


def model_config(ndim):
    c = adopted_config(ndim)
    c["zero_response_constraint"] = dict(
        version=VERSION,
        formula="F(u) - P_high F(0)",
        cutoff=ARCHITECTURES[ndim][1],
        high_mask="max_j abs(integer_frequency_j) >= existing_modes",
        zero_mode_preserved=True,
        physical_coefficients_supplied=False,
        additional_trainable_parameters=0,
    )
    return c


class HighFrequencyConsistentSGNO(nn.Module):
    def __init__(self, predictor, ndim):
        super().__init__()
        self.predictor, self.ndim = (predictor, ndim)
        n = SPATIAL_SIZES[ndim]
        self.spatial_shape = (n,) * ndim
        self.cutoff = ARCHITECTURES[ndim][1]
        f = [torch.fft.fftfreq(n, d=1 / n) for _ in range(ndim - 1)]
        f += [torch.fft.rfftfreq(n, d=1 / n)]
        grid = torch.meshgrid(*f, indexing="ij")
        self.register_buffer(
            "high_mask",
            torch.stack([g.abs() >= self.cutoff for g in grid]).any(0),
            persistent=False,
        )
        self._eval_correction = None

    def highpass(self, x):
        dims = tuple(range(-self.ndim, 0))
        return torch.fft.irfftn(
            torch.fft.rfftn(x.float(), dim=dims) * self.high_mask,
            s=self.spatial_shape,
            dim=dims,
        ).to(x.dtype)

    def train(self, mode=True):
        self._eval_correction = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._eval_correction = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._eval_correction = None
        return super()._load_from_state_dict(*args, **kwargs)

    def correction(self, u):
        if not self.training and (not torch.is_grad_enabled()):
            if self._eval_correction is None:
                self._eval_correction = self.highpass(
                    self.predictor(torch.zeros_like(u[:1]))
                ).detach()
            return self._eval_correction
        return self.highpass(self.predictor(torch.zeros_like(u[:1])))

    def forward(self, u):
        return self.predictor(u) - self.correction(u)

    def predict_next(self, u):
        return self(u)

    def learned_state_propagation(self):
        r = self.predictor.learned_state_propagation()
        r.update(
            zero_input_high_band_removed=True,
            fixed_cutoff=self.cutoff,
            scope="complete filtered prediction minus its high-frequency zero-input response",
        )
        return r


def build_model(ndim, seed=0):
    if seed != 0 or ndim not in SPATIAL_SIZES:
        raise ValueError("one fixed all14 seed0 candidate")
    return HighFrequencyConsistentSGNO(build_adopted(ndim, seed), ndim)


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
        inherited_initialization="exact adopted backbone and RNG; no added random parameters",
        prediction="F(u) - P_high F(0), jointly differentiated during training",
        evaluation="zero-input correction cached once per frozen model evaluation",
    )
