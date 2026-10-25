"""
Unit tests confirming exact objective alignment between Ridge and SGD (alpha_Ridge = m * alpha_SGD).
"""

import numpy as np
from sklearn.linear_model import Ridge
import pytest

from sgdrepro.ridge import fit_ridge_reference


def test_alpha_ridge_gradient_zero():
    """
    Verify that the analytical gradient of the SGD regularized loss function:
      grad = (1 / m) * X^T (X * w + b - y) + alpha * w
    evaluates to zero (< 1e-8) at the closed-form Ridge solution with alpha_Ridge = m * alpha_SGD.
    """
    rng = np.random.default_rng(42)
    m, n = 200, 10
    alpha_sgd = 1e-3

    X = rng.standard_normal((m, n))
    y = rng.standard_normal(m)

    # Standardize X and center y to verify with fit_intercept=True
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X_std = (X - x_mean) / x_std
    y_std = (y - np.mean(y)) / np.std(y)

    w_ridge, b_ridge = fit_ridge_reference(X_std, y_std, alpha_sgd=alpha_sgd)

    # Compute SGD objective gradient w.r.t w
    # L(w, b) = (1 / (2m)) ||X w + b - y||^2 + (alpha / 2) ||w||^2
    # grad_w = (1 / m) X^T (X w + b - y) + alpha * w
    residuals = X_std.dot(w_ridge) + b_ridge - y_std
    grad_w = (1.0 / m) * X_std.T.dot(residuals) + alpha_sgd * w_ridge

    grad_norm = np.linalg.norm(grad_w)
    assert grad_norm < 1e-8, f"Gradient norm at Ridge solution is {grad_norm}, expected < 1e-8"

    # Also check gradient w.r.t b: (1 / m) sum(residuals)
    grad_b = (1.0 / m) * np.sum(residuals)
    assert abs(grad_b) < 1e-8, f"Bias gradient at Ridge solution is {grad_b}, expected < 1e-8"
