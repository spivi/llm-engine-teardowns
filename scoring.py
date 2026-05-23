"""Parse and score LLM responses against ground truth.

Used by:
- the runner (to classify each row)
- the agent provider (to give feedback between attempts and to detect when the
  model already produced an acceptable answer mid-loop)
- the analysis layer (for the markdown reports)

Conventions: every function returns (value, error_type) or a small structured
result. error_type values come from `constants`.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from constants import (
    ERROR_TYPE_FAILED_TO_PARSE,
    ERROR_TYPE_INFEASIBLE_SOLUTION,
    ERROR_TYPE_NONE,
    ERROR_TYPE_SUBOPTIMAL_SOLUTION,
    ERROR_TYPE_WRONG_SHAPE,
    MARKOWITZ_OPTIMALITY_REL_TOL,
    TSP_TW_OPTIMALITY_ABS_TOL,
)


_ANSWER_MARKER_RE = re.compile(r"ANSWER\s*:\s*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# JSON answer extraction
# ---------------------------------------------------------------------------

def _last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find the last balanced top-level JSON object in `text`."""
    if not text:
        return None
    # First try: strict — text is exactly a JSON object.
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try: look after the last ANSWER: marker.
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if matches:
        tail = text[matches[-1].end():].strip()
        tail = _strip_code_fences(tail)
        try:
            obj = json.loads(tail)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        obj = _balanced_object_search(tail)
        if obj is not None:
            return obj

    # Last resort: balanced search across the whole text.
    return _balanced_object_search(text)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def _balanced_object_search(text: str) -> Optional[Dict[str, Any]]:
    """Find the last balanced {...} substring that parses as a dict."""
    last_obj = None
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start: i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        last_obj = obj
                except json.JSONDecodeError:
                    pass
                start = -1
    return last_obj


# ---------------------------------------------------------------------------
# Per-problem parsing
# ---------------------------------------------------------------------------

def parse_markowitz_answer(raw_text: str) -> Tuple[Optional[List[float]], str]:
    """Extract a `weights` list of 10 floats from the model's response."""
    obj = _last_json_object(raw_text)
    if obj is None:
        return None, ERROR_TYPE_FAILED_TO_PARSE
    if "weights" not in obj:
        return None, ERROR_TYPE_WRONG_SHAPE
    w = obj["weights"]
    if not isinstance(w, list):
        return None, ERROR_TYPE_WRONG_SHAPE
    if len(w) != 10:
        return None, ERROR_TYPE_WRONG_SHAPE
    try:
        weights = [float(x) for x in w]
    except (TypeError, ValueError):
        return None, ERROR_TYPE_WRONG_SHAPE
    if any(not math.isfinite(x) for x in weights):
        return None, ERROR_TYPE_WRONG_SHAPE
    return weights, ERROR_TYPE_NONE


def parse_tsp_tw_answer(raw_text: str) -> Tuple[Optional[List[int]], str]:
    """Extract a `tour` list of 13 ints from the model's response."""
    obj = _last_json_object(raw_text)
    if obj is None:
        return None, ERROR_TYPE_FAILED_TO_PARSE
    if "tour" not in obj:
        return None, ERROR_TYPE_WRONG_SHAPE
    t = obj["tour"]
    if not isinstance(t, list):
        return None, ERROR_TYPE_WRONG_SHAPE
    if len(t) != 13:
        return None, ERROR_TYPE_WRONG_SHAPE
    try:
        tour = [int(x) for x in t]
    except (TypeError, ValueError):
        return None, ERROR_TYPE_WRONG_SHAPE
    return tour, ERROR_TYPE_NONE


# ---------------------------------------------------------------------------
# Feasibility / objective evaluation against the stored instance
# ---------------------------------------------------------------------------

def evaluate_markowitz(
    weights: Optional[List[float]],
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    if weights is None:
        return _empty_eval(ERROR_TYPE_FAILED_TO_PARSE)

    w = np.asarray(weights, dtype=float)
    Sigma = np.asarray(instance["cov_matrix"], dtype=float)
    mu = np.asarray(instance["returns"], dtype=float)
    target = float(instance["target_return"])

    violations = []
    if abs(w.sum() - 1.0) > 1e-3:
        violations.append(f"sum(weights) = {w.sum():.6f}, expected ≈ 1")
    if (w < -1e-6).any():
        bad = int(np.argmin(w))
        violations.append(f"weights[{bad}] = {w[bad]:.6f} < 0 (short selling not allowed)")
    achieved_return = float(mu @ w)
    if achieved_return < target - 1e-6:
        violations.append(
            f"expected return {achieved_return:.6f} < target {target:.6f}"
        )

    variance = float(w @ Sigma @ w)
    optimal_variance = float(instance["optimal_variance"])
    rel_gap = (variance - optimal_variance) / max(abs(optimal_variance), 1e-12)

    if violations:
        return {
            "feasible": False,
            "feasibility_message": "; ".join(violations),
            "variance": variance,
            "expected_return": achieved_return,
            "optimality_gap": None,
            "error_type": ERROR_TYPE_INFEASIBLE_SOLUTION,
        }

    if rel_gap <= MARKOWITZ_OPTIMALITY_REL_TOL:
        et = ERROR_TYPE_NONE
    else:
        et = ERROR_TYPE_SUBOPTIMAL_SOLUTION

    return {
        "feasible": True,
        "feasibility_message": "",
        "variance": variance,
        "expected_return": achieved_return,
        "optimality_gap": rel_gap,
        "error_type": et,
    }


def evaluate_tsp_tw(
    tour: Optional[List[int]],
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    if tour is None:
        return _empty_eval(ERROR_TYPE_FAILED_TO_PARSE)

    n = len(instance["coordinates"])
    coords = instance["coordinates"]
    windows = instance["time_windows"]
    service_time = int(instance["service_time"])
    speed = float(instance["speed"])

    if tour[0] != 0 or tour[-1] != 0:
        return {
            "feasible": False,
            "feasibility_message": (
                f"tour starts at {tour[0]} and ends at {tour[-1]}; expected 0..0"
            ),
            "total_time": None,
            "optimality_gap": None,
            "error_type": ERROR_TYPE_INFEASIBLE_SOLUTION,
        }
    visited = set(tour[:-1])
    if visited != set(range(n)):
        missing = sorted(set(range(n)) - visited)
        extra = sorted(visited - set(range(n)))
        msg = f"tour does not visit every city exactly once (missing {missing}, extra {extra})"
        return {
            "feasible": False,
            "feasibility_message": msg,
            "total_time": None,
            "optimality_gap": None,
            "error_type": ERROR_TYPE_INFEASIBLE_SOLUTION,
        }
    if len(tour) - 1 != n:
        return {
            "feasible": False,
            "feasibility_message": f"tour visits {len(tour) - 1} positions; expected {n} + return-to-depot",
            "total_time": None,
            "optimality_gap": None,
            "error_type": ERROR_TYPE_INFEASIBLE_SOLUTION,
        }

    # Simulate the tour with the spec's integer arithmetic.
    t = 0.0
    for k in range(1, len(tour)):
        prev, curr = tour[k - 1], tour[k]
        travel = round(
            math.hypot(coords[prev][0] - coords[curr][0],
                       coords[prev][1] - coords[curr][1]) / speed
        )
        srv = service_time if prev != 0 else 0
        t = t + travel + srv
        lo, hi = windows[curr]
        if t > hi + 1e-6:
            return {
                "feasible": False,
                "feasibility_message": (
                    f"arrives at city {curr} at time {t} after window close {hi}"
                ),
                "total_time": None,
                "optimality_gap": None,
                "error_type": ERROR_TYPE_INFEASIBLE_SOLUTION,
            }
        if t < lo:
            t = float(lo)  # vehicle waits until window opens

    total_time = int(round(t))
    optimal_total_time = int(instance["optimal_total_time"])
    time_diff = total_time - optimal_total_time
    if abs(time_diff) <= TSP_TW_OPTIMALITY_ABS_TOL:
        et = ERROR_TYPE_NONE
    else:
        et = ERROR_TYPE_SUBOPTIMAL_SOLUTION

    return {
        "feasible": True,
        "feasibility_message": "",
        "total_time": total_time,
        "optimality_gap": (total_time - optimal_total_time) / max(optimal_total_time, 1),
        "error_type": et,
    }


def _empty_eval(error_type: str) -> Dict[str, Any]:
    return {
        "feasible": False,
        "feasibility_message": "answer not parseable",
        "variance": None,
        "expected_return": None,
        "total_time": None,
        "optimality_gap": None,
        "error_type": error_type,
    }


# ---------------------------------------------------------------------------
# Dispatch by subproblem
# ---------------------------------------------------------------------------

def evaluate(raw_text: str, instance: Dict[str, Any]) -> Dict[str, Any]:
    sub = instance["subproblem"]
    if sub == "markowitz":
        parsed, parse_err = parse_markowitz_answer(raw_text)
        if parsed is None:
            return {**_empty_eval(parse_err), "parsed_answer": None}
        ev = evaluate_markowitz(parsed, instance)
        return {**ev, "parsed_answer": parsed}
    if sub == "tsp_tw":
        parsed, parse_err = parse_tsp_tw_answer(raw_text)
        if parsed is None:
            return {**_empty_eval(parse_err), "parsed_answer": None}
        ev = evaluate_tsp_tw(parsed, instance)
        return {**ev, "parsed_answer": parsed}
    raise ValueError(f"unknown subproblem: {sub!r}")


def feedback_message(eval_result: Dict[str, Any]) -> str:
    """Produce a short feedback message for the agent's next attempt."""
    et = eval_result.get("error_type", "")
    if et == ERROR_TYPE_FAILED_TO_PARSE:
        return "Your previous response could not be parsed as JSON with the required field. Please return only the JSON object on its own line."
    if et == ERROR_TYPE_WRONG_SHAPE:
        return "Your previous answer had the wrong shape (e.g., missing key or wrong list length). Please return exactly the expected JSON shape."
    if et == ERROR_TYPE_INFEASIBLE_SOLUTION:
        return f"Your previous answer is infeasible: {eval_result.get('feasibility_message', '')}. Try again."
    if et == ERROR_TYPE_SUBOPTIMAL_SOLUTION:
        gap = eval_result.get("optimality_gap", 0.0)
        return (
            f"Your previous answer is feasible but suboptimal "
            f"(objective relative gap {gap:.4f} above optimal). Try to improve it."
        )
    return ""
