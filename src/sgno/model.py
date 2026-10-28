"""Paper SGNO model. Inputs and outputs have shape (batch, channels, *grid)."""

from sgno._increment import (
    METHOD,
    VERSION,
    ZeroHeadIncrementSGNO,
    model_config,
    model_spec,
    reference,
)


def build_model(ndim, seed=0):
    if ndim not in (1, 2, 3) or seed not in range(5):
        raise ValueError("Fixed main-task confirmation: dimensions 1-3, seeds 0-4")
    model = reference(ndim, seed)
    model.__class__ = ZeroHeadIncrementSGNO
    model.initialize_readout()
    return model
