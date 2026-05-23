"""Reference Markowitz portfolio solver via CVXPY.

Problem:
    minimize   w^T Sigma w
    subject to mu^T w >= target_return
               sum(w) == 1
               w >= 0
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import cvxpy as cp
import numpy as np


_STATUS_MAP = {
    cp.OPTIMAL: "optimal",
    cp.OPTIMAL_INACCURATE: "optimal",
    cp.INFEASIBLE: "infeasible",
    cp.INFEASIBLE_INACCURATE: "infeasible",
    cp.UNBOUNDED: "unbounded",
    cp.UNBOUNDED_INACCURATE: "unbounded",
}


def solve(returns: List[float], cov_matrix: List[List[float]], target_return: float) -> Dict[str, Any]:
    """Solve the Markowitz problem with CVXPY (CLARABEL backend)."""
    mu = np.asarray(returns, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)
    n = mu.shape[0]
    assert Sigma.shape == (n, n), f"shape mismatch: mu has {n} entries, Sigma is {Sigma.shape}"

    w = cp.Variable(n)
    constraints = [
        mu @ w >= float(target_return),
        cp.sum(w) == 1.0,
        w >= 0,
    ]
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), constraints)

    t0 = time.perf_counter()
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as e:
        return {
            "status": "error",
            "weights": None,
            "variance": None,
            "expected_return": None,
            "solve_ms": (time.perf_counter() - t0) * 1000.0,
            "message": f"cvxpy SolverError: {e}",
        }
    solve_ms = (time.perf_counter() - t0) * 1000.0

    status = _STATUS_MAP.get(problem.status, "error")
    if status != "optimal" or w.value is None:
        return {
            "status": status,
            "weights": None,
            "variance": None,
            "expected_return": None,
            "solve_ms": solve_ms,
            "message": f"cvxpy status: {problem.status}",
        }

    weights = np.asarray(w.value)
    # Clean numerical noise: clip tiny negatives, renormalize.
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    variance = float(weights @ Sigma @ weights)
    expected_return = float(mu @ weights)
    return {
        "status": "optimal",
        "weights": weights.tolist(),
        "variance": variance,
        "expected_return": expected_return,
        "solve_ms": solve_ms,
        "message": "",
    }
