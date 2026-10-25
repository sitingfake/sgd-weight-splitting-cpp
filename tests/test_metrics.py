"""
Unit tests comparing internal metrics to scikit-learn metrics.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as sk_r2_score
import pytest

from sgdrepro.metrics import rmse, mae, r2_score, sgd_objective_loss


def test_metrics_parity_with_sklearn():
    """Verify RMSE, MAE, and R^2 match scikit-learn down to floating point precision."""
    rng = np.random.default_rng(42)
    y_true = rng.standard_normal(500) * 100.0 + 250.0
    y_pred = y_true + rng.standard_normal(500) * 15.0

    # RMSE
    sk_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    our_rmse = rmse(y_true, y_pred)
    np.testing.assert_allclose(our_rmse, sk_rmse, rtol=1e-14)

    # MAE
    sk_mae = mean_absolute_error(y_true, y_pred)
    our_mae = mae(y_true, y_pred)
    np.testing.assert_allclose(our_mae, sk_mae, rtol=1e-14)

    # R^2
    sk_r2 = sk_r2_score(y_true, y_pred)
    our_r2 = r2_score(y_true, y_pred)
    np.testing.assert_allclose(our_r2, sk_r2, rtol=1e-14)


def test_sgd_objective_loss_convexity():
    """Verify regularized loss evaluates correctly and increases when deviating from optimum."""
    rng = np.random.default_rng(123)
    X = rng.standard_normal((100, 5))
    w_star = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
    y = X.dot(w_star)

    loss_star = sgd_objective_loss(X, y, w_star, b=0.0, alpha=1e-4)
    loss_perturbed = sgd_objective_loss(X, y, w_star + 0.1, b=0.0, alpha=1e-4)

    assert loss_star >= 0.0
    assert loss_perturbed > loss_star
