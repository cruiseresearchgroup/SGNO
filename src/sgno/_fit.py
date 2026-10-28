"""Training-pair constrained affine Fourier least squares."""

import numpy as np


def low_mask(shape, modes):
    f = [np.fft.fftfreq(n, d=1 / n) for n in shape[:-1]]
    f += [np.fft.rfftfreq(shape[-1], d=1 / shape[-1])]
    grid = np.meshgrid(*f, indexing="ij")
    return np.logical_and.reduce([np.abs(g) < modes for g in grid])


def fit(train, modes):
    """Use every adjacent pair in the supplied TRAIN array; no test access."""
    assert train.ndim in (4, 5, 6) and train.shape[2] == 1
    shape = train.shape[3:]
    axes = tuple(range(-len(shape), 0))
    spectral_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    sx = np.zeros(spectral_shape, np.complex128)
    sy = np.zeros_like(sx)
    sxy = np.zeros_like(sx)
    sxx = np.zeros(spectral_shape, np.float64)
    count = 0
    for ic in range(len(train)):
        h = np.fft.rfftn(train[ic, :, 0].astype(np.float64), axes=axes, norm="ortho")
        x, y = (h[:-1], h[1:])
        sx += x.sum(0)
        sy += y.sum(0)
        sxx += (x.real * x.real + x.imag * x.imag).sum(0)
        sxy += (x.conj() * y).sum(0)
        count += len(x)
    mask = low_mask(shape, modes)
    variance = np.maximum(np.where(mask, sxx - abs(sx) ** 2 / count, sxx), 0)
    cross = np.where(mask, sxy - sx.conj() * sy / count, sxy)
    floor = max(float(variance.max()) * 1e-12, np.finfo(np.float64).tiny)
    observed = variance > floor
    raw = np.ones(spectral_shape, np.complex128)
    raw[observed] = cross[observed] / variance[observed]
    gain = raw / np.maximum(1.0, abs(raw))
    bias = np.where(mask, (sy - gain * sx) / count, 0.0)
    report = dict(
        training_ics=len(train),
        pairs=count,
        modes=modes,
        variance_floor=floor,
        observed_modes=int(observed.sum()),
        stored_multiplier_elements=gain.size,
        stored_bias_elements=int(mask.sum()),
        clipped_modes=int((abs(raw) > 1).sum()),
        max_raw_gain=float(abs(raw).max()),
        max_effective_gain=float(abs(gain).max()),
        no_test_data=True,
    )
    return (gain, bias, report)


def apply(x, gain, bias):
    shape = x.shape[2:]
    axes = tuple(range(-len(shape), 0))
    h = np.fft.rfftn(x.astype(np.float64), axes=axes, norm="ortho")
    return np.fft.irfftn(h * gain + bias, s=shape, axes=axes, norm="ortho")
