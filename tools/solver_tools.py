"""Constrained solver API exposed to the agentic provider.

Two tools, both returning a uniform-shape dict:
    {"status": "optimal" | "infeasible" | "unbounded" | "error",
     "x": list[float] | None,
     "objective": float | None,
     "message": str}

Validation is fail-fast with informative messages — the LLM reads these
to self-correct on retries. Validation order is cheap → expensive:
    1. coerce + NaN/Inf check
    2. shape consistency
    3. (QP only) symmetry of P
    4. (QP only) PSD via smallest eigenvalue
    5. bounds consistency
    6. solve
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
from scipy import optimize as _spo

_PSD_TOL = 1e-8
_SYMMETRY_TOL = 1e-8


def _err(msg: str) -> Dict[str, Any]:
    return {"status": "error", "x": None, "objective": None, "message": msg}


def _to_2d(arr, name: str) -> Optional[np.ndarray]:
    try:
        a = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} could not be cast to float array: {e}")
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (got {a.ndim}-D, shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Inf")
    return a


def _to_1d(arr, name: str) -> Optional[np.ndarray]:
    try:
        a = np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} could not be cast to float array: {e}")
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (got {a.ndim}-D, shape {a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or Inf")
    return a


def _validate_bounds(bounds, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.full(n, -np.inf), np.full(n, np.inf)
    if len(bounds) != n:
        raise ValueError(
            f"bounds length {len(bounds)} != number of variables {n}"
        )
    lb = np.empty(n)
    ub = np.empty(n)
    for i, pair in enumerate(bounds):
        if pair is None:
            lb[i], ub[i] = -np.inf, np.inf
            continue
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(
                f"bounds[{i}] must be a (lo, hi) pair (got {pair!r})"
            )
        lo, hi = pair
        lb[i] = -np.inf if lo is None else float(lo)
        ub[i] = np.inf if hi is None else float(hi)
        if not (np.isfinite(lb[i]) or lb[i] == -np.inf):
            raise ValueError(f"bounds[{i}][0]={lo!r} is not a finite number or None")
        if not (np.isfinite(ub[i]) or ub[i] == np.inf):
            raise ValueError(f"bounds[{i}][1]={hi!r} is not a finite number or None")
        if lb[i] > ub[i]:
            raise ValueError(
                f"bounds[{i}] infeasible: lb={lb[i]} > ub={ub[i]}"
            )
    return lb, ub


# ---------------------------------------------------------------------------
# solve_qp
# ---------------------------------------------------------------------------

def solve_qp(
    P: Sequence[Sequence[float]],
    q: Sequence[float],
    G: Optional[Sequence[Sequence[float]]] = None,
    h: Optional[Sequence[float]] = None,
    A: Optional[Sequence[Sequence[float]]] = None,
    b: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, Any]:
    """Solve a convex quadratic program.

        minimize   (1/2) x^T P x + q^T x
        subject to G x <= h
                   A x == b
                   bounds[i][0] <= x[i] <= bounds[i][1]

    P must be symmetric positive semi-definite (within tolerance).
    """
    try:
        P_arr = _to_2d(P, "P")
        q_arr = _to_1d(q, "q")
    except ValueError as e:
        return _err(str(e))

    n = q_arr.shape[0]
    if P_arr.shape != (n, n):
        return _err(
            f"shape mismatch: P is {P_arr.shape} but q has length {n}; "
            f"expected P shape ({n}, {n})"
        )

    asym = float(np.max(np.abs(P_arr - P_arr.T)))
    if asym > _SYMMETRY_TOL:
        return _err(
            f"P is not symmetric: max |P - P.T| = {asym:.3e} (tolerance {_SYMMETRY_TOL}). "
            "QP requires symmetric P. Pass 0.5*(P + P.T) if asymmetry was unintentional."
        )

    try:
        eigmin = float(np.linalg.eigvalsh(P_arr)[0])
    except np.linalg.LinAlgError as e:
        return _err(f"could not compute eigenvalues of P: {e}")
    if eigmin < -_PSD_TOL:
        return _err(
            f"P is not positive semi-definite: smallest eigenvalue = {eigmin:.3e} "
            f"(tolerance -{_PSD_TOL}). QP requires P psd. "
            "Consider adding a regularizer P + eps*I, or check signs."
        )

    try:
        if G is not None or h is not None:
            if G is None or h is None:
                return _err("G and h must be provided together (or both omitted)")
            G_arr = _to_2d(G, "G")
            h_arr = _to_1d(h, "h")
            if G_arr.shape[1] != n:
                return _err(f"shape mismatch: G has {G_arr.shape[1]} columns, expected {n}")
            if G_arr.shape[0] != h_arr.shape[0]:
                return _err(
                    f"shape mismatch: G is {G_arr.shape} but h has length {h_arr.shape[0]}; "
                    f"need G.shape[0] == len(h)"
                )
        else:
            G_arr = h_arr = None

        if A is not None or b is not None:
            if A is None or b is None:
                return _err("A and b must be provided together (or both omitted)")
            A_arr = _to_2d(A, "A")
            b_arr = _to_1d(b, "b")
            if A_arr.shape[1] != n:
                return _err(f"shape mismatch: A has {A_arr.shape[1]} columns, expected {n}")
            if A_arr.shape[0] != b_arr.shape[0]:
                return _err(
                    f"shape mismatch: A is {A_arr.shape} but b has length {b_arr.shape[0]}; "
                    f"need A.shape[0] == len(b)"
                )
        else:
            A_arr = b_arr = None

        lb, ub = _validate_bounds(bounds, n)
    except ValueError as e:
        return _err(str(e))

    # Build the CVXPY problem. Wrap P with psd_wrap because we already checked PSD.
    x = cp.Variable(n)
    objective = 0.5 * cp.quad_form(x, cp.psd_wrap(P_arr)) + q_arr @ x
    constraints = []
    if G_arr is not None:
        constraints.append(G_arr @ x <= h_arr)
    if A_arr is not None:
        constraints.append(A_arr @ x == b_arr)
    for i in range(n):
        if np.isfinite(lb[i]):
            constraints.append(x[i] >= lb[i])
        if np.isfinite(ub[i]):
            constraints.append(x[i] <= ub[i])

    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as e:
        return _err(f"cvxpy SolverError: {e}")

    status = problem.status
    if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and x.value is not None:
        return {
            "status": "optimal",
            "x": [float(v) for v in np.asarray(x.value)],
            "objective": float(problem.value),
            "message": "",
        }
    if status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        return {
            "status": "infeasible",
            "x": None,
            "objective": None,
            "message": (
                "infeasible: constraints have no solution. Common causes: "
                "contradictory equality constraints, lb > ub on some variable, "
                "or A x = b inconsistent with G x <= h."
            ),
        }
    if status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        return {
            "status": "unbounded",
            "x": None,
            "objective": None,
            "message": (
                "objective decreases without limit. If P has zero eigenvalues and "
                "q points in a free direction, add bounds or extra constraints."
            ),
        }
    return _err(f"cvxpy returned status={status}; objective={problem.value}")


# ---------------------------------------------------------------------------
# solve_milp
# ---------------------------------------------------------------------------

# scipy.optimize.milp termination message → uniform status mapping.
# HiGHS status codes (from scipy docs):
#   0: optimal | success
#   1: iteration / time / node limit
#   2: infeasible
#   3: unbounded
#   4: other
def solve_milp(
    c: Sequence[float],
    A_ub: Optional[Sequence[Sequence[float]]] = None,
    b_ub: Optional[Sequence[float]] = None,
    A_eq: Optional[Sequence[Sequence[float]]] = None,
    b_eq: Optional[Sequence[float]] = None,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    integrality: Optional[Sequence[int]] = None,
    time_limit_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve a mixed-integer linear program.

        minimize   c^T x
        subject to A_ub x <= b_ub
                   A_eq x == b_eq
                   bounds[i][0] <= x[i] <= bounds[i][1]
                   x[i] integer if integrality[i] == 1, else continuous
    """
    try:
        c_arr = _to_1d(c, "c")
    except ValueError as e:
        return _err(str(e))
    n = c_arr.shape[0]

    try:
        if A_ub is not None or b_ub is not None:
            if A_ub is None or b_ub is None:
                return _err("A_ub and b_ub must be provided together (or both omitted)")
            A_ub_arr = _to_2d(A_ub, "A_ub")
            b_ub_arr = _to_1d(b_ub, "b_ub")
            if A_ub_arr.shape[1] != n:
                return _err(
                    f"shape mismatch: A_ub has {A_ub_arr.shape[1]} columns, expected {n}"
                )
            if A_ub_arr.shape[0] != b_ub_arr.shape[0]:
                return _err(
                    f"shape mismatch: A_ub is {A_ub_arr.shape} but b_ub has length "
                    f"{b_ub_arr.shape[0]}"
                )
        else:
            A_ub_arr = b_ub_arr = None

        if A_eq is not None or b_eq is not None:
            if A_eq is None or b_eq is None:
                return _err("A_eq and b_eq must be provided together (or both omitted)")
            A_eq_arr = _to_2d(A_eq, "A_eq")
            b_eq_arr = _to_1d(b_eq, "b_eq")
            if A_eq_arr.shape[1] != n:
                return _err(
                    f"shape mismatch: A_eq has {A_eq_arr.shape[1]} columns, expected {n}"
                )
            if A_eq_arr.shape[0] != b_eq_arr.shape[0]:
                return _err(
                    f"shape mismatch: A_eq is {A_eq_arr.shape} but b_eq has length "
                    f"{b_eq_arr.shape[0]}"
                )
        else:
            A_eq_arr = b_eq_arr = None

        lb, ub = _validate_bounds(bounds, n)
    except ValueError as e:
        return _err(str(e))

    if integrality is not None:
        if len(integrality) != n:
            return _err(
                f"integrality length {len(integrality)} != number of variables {n}"
            )
        intg = np.asarray(integrality, dtype=int)
        if not np.all((intg == 0) | (intg == 1)):
            return _err(
                "integrality entries must be 0 (continuous) or 1 (integer); "
                f"got {sorted(set(int(v) for v in intg))}"
            )
    else:
        intg = np.zeros(n, dtype=int)

    # Build scipy LinearConstraints.
    constraints = []
    if A_ub_arr is not None:
        constraints.append(_spo.LinearConstraint(A_ub_arr, -np.inf, b_ub_arr))
    if A_eq_arr is not None:
        constraints.append(_spo.LinearConstraint(A_eq_arr, b_eq_arr, b_eq_arr))
    spo_bounds = _spo.Bounds(lb=lb, ub=ub)

    options: Dict[str, Any] = {}
    if time_limit_s is not None:
        options["time_limit"] = float(time_limit_s)

    try:
        result = _spo.milp(
            c=c_arr,
            constraints=constraints,
            bounds=spo_bounds,
            integrality=intg,
            options=options or None,
        )
    except Exception as e:
        return _err(f"scipy.optimize.milp raised: {type(e).__name__}: {e}")

    # result.status: 0=optimal, 1=iter/time/node limit, 2=infeasible, 3=unbounded, 4=other
    if result.status == 0 and result.x is not None:
        return {
            "status": "optimal",
            "x": [float(v) for v in np.asarray(result.x)],
            "objective": float(result.fun),
            "message": "",
        }
    if result.status == 2:
        return {
            "status": "infeasible",
            "x": None,
            "objective": None,
            "message": (
                "infeasible: constraints have no solution. Common causes: "
                "contradictory equality constraints, lb > ub on some variable, "
                "or integrality requirements making the LP relaxation feasible region empty."
            ),
        }
    if result.status == 3:
        return {
            "status": "unbounded",
            "x": None,
            "objective": None,
            "message": (
                "objective is unbounded below. Add bounds on unbounded variables "
                "or check the sign of c."
            ),
        }
    if result.status == 1:
        return _err(
            f"milp hit a limit (iteration / time / node): {result.message}. "
            f"Increase time_limit_s or simplify the formulation."
        )
    return _err(
        f"milp returned status={result.status}: {result.message}"
    )
