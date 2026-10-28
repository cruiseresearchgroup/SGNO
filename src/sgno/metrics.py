"""Paper error conventions; no filtering of difficult trajectories or seeds."""

import numpy as np


def field_diagnostics(prediction, reference):
    """Per-field diagnostics for arrays shaped (batch, channel, *grid).

    Amplitude and amplitude-weighted phase contributions sum to normalized
    squared field error. Correlation is the normalized inner product, with no
    additional mean subtraction. Undefined ratios remain non-finite.
    """
    a = np.asarray(prediction, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or a.ndim < 3:
        raise ValueError("Expected matching (batch, channel, *grid) fields")
    spatial = tuple(range(2, a.ndim))
    reduce = tuple(range(1, a.ndim))
    with np.errstate(all="ignore"):
        ah = np.fft.rfftn(a, axes=spatial, norm="ortho")
        bh = np.fft.rfftn(b, axes=spatial, norm="ortho")
        weights = np.full(ah.shape[-1], 2.0)
        weights[0] = 1.0
        if a.shape[-1] % 2 == 0:
            weights[-1] = 1.0
        reference_energy = np.sum(b * b, axis=reduce)
        prediction_energy = np.sum(a * a, axis=reduce)
        amplitude = np.sum((np.abs(ah) - np.abs(bh)) ** 2 * weights, axis=reduce)
        total = np.sum(np.abs(ah - bh) ** 2 * weights, axis=reduce)
        return {
            "amplitude_squared": amplitude / reference_energy,
            "phase_squared": np.maximum(total - amplitude, 0.0) / reference_energy,
            "total_squared": total / reference_energy,
            "squared_norm_ratio": prediction_energy / reference_energy,
            "field_correlation": np.sum(a * b, axis=reduce)
            / np.sqrt(prediction_energy * reference_energy),
        }


def relative_rms(prediction, reference, ndim):
    a, b = np.asarray(prediction), np.asarray(reference)
    if ndim == 3:
        delta = (a.astype(np.float32) - b.astype(np.float32)).astype(np.float64)
        return np.linalg.norm(delta.reshape(len(a), -1), axis=1) / (
            np.linalg.norm(b.astype(np.float64).reshape(len(b), -1), axis=1) + 1e-12
        )
    delta = a.astype(np.float64) - b.astype(np.float64)
    axes = tuple(range(1, a.ndim))
    return np.sqrt(np.mean(delta**2, axis=axes)) / np.maximum(
        np.sqrt(np.mean(b.astype(np.float64) ** 2, axis=axes)), 1e-12
    )


def summarize(errors):
    errors = np.asarray(errors, dtype=np.float64)
    with np.errstate(all="ignore"):
        curve = errors.mean(axis=0)
        result = {
            "test_trajectories": len(errors),
            "steps": len(curve),
            "nonfinite_trajectory_steps": int((~np.isfinite(errors)).sum()),
        }
        for start, stop in ((1, 100), (101, 200)):
            if len(curve) >= stop:
                result[f"gmean_{start}_{stop}"] = float(
                    np.exp(np.log(curve[start - 1 : stop] + 1e-12).mean())
                )
        result["endpoint_nrmse"] = float(curve[-1])
        if len(curve) >= 200:
            time = np.arange(101, 201, dtype=np.float64)
            time -= time.mean()
            y = curve[100:200]
            result["late_absolute_slope"] = float(
                np.dot(time, y - y.mean()) / np.dot(time, time)
            )
            result["late_log_slope"] = float(
                np.dot(time, np.log(y) - np.log(y).mean()) / np.dot(time, time)
            )
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
