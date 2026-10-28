"""Zero-readout SGNO correction coupled to the fitted state map."""

import torch
from sgno._seed import build_model as reference
from sgno._mean import MeanPreservingSGNO
from sgno._mean import model_config as parent_config
from sgno.sgno import _make_grid_1d, _make_grid_2d, _make_grid_3d

METHOD = "sgno_zero_head_increment_v1"
VERSION = "affine-plus-filtered-zero-head-sgno-increment-v1"


def model_config(ndim):
    config = parent_config(ndim)
    config["physical_propagation"].update(
        version=VERSION,
        formula="P0(u)+Q[L(Q(u))+c+alpha*G(Delta(Q(u))-P_high Delta(0))]",
        initialization="Same training-only affine fit; zero final decoder; alpha=1",
    )
    config["mean_preservation"].update(
        residual_weight_initial=1.0, formula=config["physical_propagation"]["formula"]
    )
    config["increment_readout"] = dict(
        version=VERSION,
        decoder="Original fc2 weight and bias initialized to zero",
        residual_weight_initial=1.0,
        original_input_skip_in_increment=False,
        filter="Original complete-update filter applied to increment only",
        zero_response="Original high-frequency zero-input subtraction",
        added_parameters=0,
        removed_parameters=0,
        task_specific_switches=False,
        initial_complete_prediction="Exactly the same fitted affine prediction",
        intervention="Increment representation and final decoder/gate initialization jointly changed",
    )
    return config


class ZeroHeadIncrementSGNO(MeanPreservingSGNO):
    @torch.no_grad()
    def initialize_readout(self):
        head = self.residual.predictor.base.core.fc2
        head.weight.zero_()
        head.bias.zero_()
        self.residual_weight.fill_(1.0)

    @torch.no_grad()
    def initialize_from_train(self, data):
        receipt = super().initialize_from_train(data)
        self.initialize_readout()
        receipt.update(
            residual_gate_initial=1.0,
            decoder_initial="zero weight and bias",
            initial_complete_prediction="unchanged training-fitted affine map",
            subsequent_training="All parameters, unchanged one-step MSE and Adam schedule",
        )
        return receipt

    def increment(self, u):
        x = u.movedim(1, -1).contiguous()
        builders = {1: _make_grid_1d, 2: _make_grid_2d, 3: _make_grid_3d}
        grid = builders[self.ndim](u.shape[0], *u.shape[2:], u.device, u.dtype)
        return (
            self.residual.predictor.base.core(x, grid)
            .squeeze(-2)
            .movedim(-1, 1)
            .contiguous()
        )

    def forward(self, u):
        axes = tuple(range(2, u.ndim))
        mean = u.mean(axes, keepdim=True)
        centered = u - mean
        residual = self.residual.predictor.filter(
            self.increment(centered)
        ) - self.residual.correction(centered)
        prediction = self.linear(centered) + self.residual_weight * residual
        return prediction - prediction.mean(axes, keepdim=True) + mean

    def learned_state_propagation(self):
        record = super().learned_state_propagation()
        record.update(
            scope="Mean-preserving affine propagation plus a filtered SGNO increment",
            residual_input_skip=False,
            decoder_initial="zero",
            residual_gate_initial=1.0,
        )
        return record


def build_model(ndim, seed=0):
    if seed != 0 or ndim not in (1, 2, 3):
        raise ValueError("One fixed all14 seed0 candidate")
    model = reference(ndim, seed)
    model.__class__ = ZeroHeadIncrementSGNO
    model.initialize_readout()
    return model


def model_spec(model):
    config = model_config(model.ndim)
    elements = sum((p.numel() for p in model.parameters()))
    real = sum((p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))
    assert (elements, real) == (config["tensor_entries"], config["real_parameters"])
    return dict(
        method=METHOD,
        version=VERSION,
        ndim=model.ndim,
        config=config,
        parameter_counts=dict(tensor_entries=elements, real_parameters=real),
        inherited_initialization="Same trunk/RNG and affine initializer; zero final decoder and unit residual weight",
        prediction=config["physical_propagation"]["formula"],
    )
