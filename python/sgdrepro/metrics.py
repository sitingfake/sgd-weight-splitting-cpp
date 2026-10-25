"""
Evaluation Metrics and Loss Functions.

Implements regression evaluation metrics (RMSE, MAE, R^2) and regularized SGD
objective loss calculations with complete mathematical parity to scikit-learn.
"""

from typing import Dict, Union
import numpy as np
import scipy.sparse as sp


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination R^2.
    R^2 = 1 - SS_res / SS_tot
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics bundle."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def sgd_objective_loss(
    X: Union[np.ndarray, sp.csr_matrix],
    y: np.ndarray,
    w: np.ndarray,
    b: float = 0.0,
    alpha: float = 1e-4,
) -> float:
    """
    Compute regularized empirical risk for linear regression:
      (1 / m) * sum( 0.5 * (X * w + b - y)^2 ) + (alpha / 2) * ||w||_2^2
    """
    m = X.shape[0]
    preds = X.dot(w) + b
    residuals = preds - y
    data_loss = 0.5 * np.mean(residuals ** 2)
    reg_loss = 0.5 * alpha * np.sum(w ** 2)
    return float(data_loss + reg_loss)


def relative_weight_difference(w1: np.ndarray, w2: np.ndarray) -> float:
    """
    Compute relative Euclidean norm discrepancy: ||w1 - w2||_2 / ||w1||_2.
    """
    norm_w1 = np.linalg.norm(w1)
    diff_norm = np.linalg.norm(w1 - w2)
    if norm_w1 == 0.0:
        return float(diff_norm)
    return float(diff_norm / norm_w1)
