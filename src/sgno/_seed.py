"""Construct the mean-preserving model from a fixed random seed."""

from sgno._mean import (
    METHOD,
    VERSION,
    MeanPreservingSGNO,
    model_config,
    model_spec,
    SPATIAL_SIZES,
)
from sgno._spec import build_reference
from sgno._filter import CompleteUpdateSGNO
from sgno._consistency import HighFrequencyConsistentSGNO


def build_model(ndim, seed=0):
    if ndim not in SPATIAL_SIZES or seed not in range(5):
        raise ValueError("Frozen model, fixed confirmation seeds 0 through 4")
    base = build_reference(ndim, seed)
    complete = CompleteUpdateSGNO(base, ndim)
    residual = HighFrequencyConsistentSGNO(complete, ndim)
    return MeanPreservingSGNO(residual, ndim)
