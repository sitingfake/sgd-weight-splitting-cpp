"""Timing and held-out evaluation shared by both research questions."""

import gc
import hashlib
from time import perf_counter_ns

import numpy as np
from scipy import sparse

from sgdrepro.metrics import compute_metrics


def timing_summary(samples, warmup):
    """Describe repeated wall-clock measurements in nanoseconds."""
    q1, q3 = np.percentile(samples, [25, 75])
    return {
        "warmup": warmup,
        "repeats": len(samples),
        "samples_ns": samples,
        "median_ns": float(np.median(samples)),
        "iqr_ns": float(q3 - q1),
    }


def benchmark_fit(make_estimator, X, y, warmup, repeats):
    """Fit fresh estimators and retain the first timed model for evaluation."""
    for _ in range(warmup):
        make_estimator().fit(X, y)

    samples = []
    first_model = None
    for _ in range(repeats):
        model = make_estimator()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = perf_counter_ns()
            model.fit(X, y)
            samples.append(perf_counter_ns() - start)
        finally:
            if gc_was_enabled:
                gc.enable()
        if first_model is None:
            first_model = model
    return first_model, timing_summary(samples, warmup)


def evaluate_holdout(X, y, w, b, warmup, repeats, target_scale=1.0, target_mean=0.0):
    """Time a shared prediction kernel and score its first timed prediction."""
    def predict():
        return (X @ w + b) * target_scale + target_mean

    for _ in range(warmup):
        predict()

    samples = []
    first_prediction = None
    for _ in range(repeats):
        start = perf_counter_ns()
        prediction = predict()
        samples.append(perf_counter_ns() - start)
        if first_prediction is None:
            first_prediction = prediction

    return {
        "metrics": compute_metrics(y, first_prediction),
        "predict_timing": timing_summary(samples, warmup),
    }


def holdout_sha256(X, y, target_scale=1.0, target_mean=0.0):
    """Identify the exact prediction input and labels shared by the models."""
    digest = hashlib.sha256(np.asarray(X.shape, dtype="<i8").tobytes())
    if sparse.issparse(X):
        digest.update(np.asarray(X.indptr, dtype="<i8").tobytes())
        digest.update(np.asarray(X.indices, dtype="<i8").tobytes())
        values = X.data
    else:
        values = X
    for part in (values, y, [target_scale, target_mean]):
        digest.update(np.asarray(part, dtype="<f8", order="C").tobytes())
    return digest.hexdigest()
