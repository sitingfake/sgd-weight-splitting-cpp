"""
Closed-Form Ridge Regression Reference Solver.

Fits Ridge regression with Cholesky factorization on standardized features
to serve as the theoretical optimal ground-truth parameter vector w_ridge.
Uses exact regularization parameter alignment: alpha_Ridge = m_train * alpha_SGD.
"""

from typing import Tuple
import numpy as np
from sklearn.linear_model import Ridge


def fit_ridge_reference(
    X_train: np.ndarray,
    y_train_std: np.ndarray,
    alpha_sgd: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    Fit Ridge regression with exact penalty alignment.

    Parameters
    ----------
    X_train : np.ndarray
        Standardized training features.
    y_train_std : np.ndarray
        Standardized training target vector.
    alpha_sgd : float
        SGD L2 regularization parameter (default: 1e-4).

    Returns
    -------
    w_ridge : np.ndarray
        Fitted weight coefficients.
    b_ridge : float
        Fitted intercept term.
    """
    m_train = X_train.shape[0]
    alpha_ridge = m_train * alpha_sgd

    # Cholesky solver computes exact closed-form solution: (X^T X + alpha I)^(-1) X^T y
    clf = Ridge(alpha=alpha_ridge, fit_intercept=True, solver="cholesky")
    clf.fit(X_train, y_train_std)

    w_ridge = np.ascontiguousarray(clf.coef_, dtype=np.float64)
    b_ridge = float(clf.intercept_)

    return w_ridge, b_ridge


def distance_to_ridge(w: np.ndarray, w_ridge: np.ndarray) -> float:
    """
    Compute relative Euclidean distance from candidate w to w_ridge:
      ||w - w_ridge||_2 / ||w_ridge||_2
    """
    norm_ref = np.linalg.norm(w_ridge)
    diff = np.linalg.norm(w - w_ridge)
    if norm_ref == 0.0:
        return float(diff)
    return float(diff / norm_ref)
