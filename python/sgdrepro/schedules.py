"""
Learning Rate Schedule Calculations.

Provides Python implementations of invscaling, bottou, pegasos, and optimal
schedules with identical parameterizations to the C++ core and scikit-learn.
"""

import numpy as np


def compute_optimal_init(alpha: float = 1e-4) -> float:
    """
    Compute optimal schedule initialization parameter for squared error loss.
      typw = sqrt(1.0 / sqrt(alpha))
      initial_eta0 = typw / max(1.0, loss.cy_gradient(1.0, -typw))
      optimal_init = 1.0 / (initial_eta0 * alpha)
    """
    typw = np.sqrt(1.0 / np.sqrt(alpha))
    # For squared error 0.5 * (p - y)^2, dloss/dp = p - y
    # At y=1.0, p=-typw: dloss = -typw - 1.0
    grad = -typw - 1.0
    initial_eta0 = typw / max(1.0, grad)
    optimal_init = 1.0 / (initial_eta0 * alpha)
    return float(optimal_init)


def get_learning_rate_py(
    schedule: str,
    t: float,
    alpha: float = 1e-4,
    eta0: float = 0.01,
    power_t: float = 0.25,
    optimal_init: float = 0.0,
) -> float:
    """Evaluate learning rate at step t (1-indexed)."""
    if schedule == "invscaling":
        return float(eta0 / (t ** power_t))
    elif schedule == "bottou":
        return float(eta0 / (1.0 + eta0 * alpha * (t - 1.0)))
    elif schedule == "pegasos":
        return float(1.0 / (alpha * t))
    elif schedule == "optimal":
        return float(1.0 / (alpha * (optimal_init + t - 1.0)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
