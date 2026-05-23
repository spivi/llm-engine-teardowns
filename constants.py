"""Shared constants for the runner, analysis, and providers.

Centralizes the error_type enumeration and optimality tolerances so the runner
(which records error_type on failure) and the analysis layer (which classifies
solutions as optimal / suboptimal / infeasible / etc.) cannot drift.

If you encounter a failure mode that does not fit, add a new value here with a
short comment explaining why, and update any analysis code that switches on
error_type.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# error_type enumeration (CSV-serializable string values)
# ---------------------------------------------------------------------------
# Decision priority (highest first) when multiple conditions could apply:
#   1. exceeded_tool_call_budget  (agent loop hit the budget without final answer)
#   2. tool_returned_error        (agent attempt ended because tool returned error / infeasible / unbounded and model gave up)
#   3. failed_to_parse            (no parseable JSON in the final response)
#   4. wrong_shape                (JSON parsed but field type / length / key wrong)
#   5. infeasible_solution        (parses, shape OK, but violates problem constraints)
#   6. suboptimal_solution        (feasible but objective worse than optimal beyond tolerance)
#   7. none                       (feasible and within optimality tolerance)
ERROR_TYPE_NONE = "none"
ERROR_TYPE_INFEASIBLE_SOLUTION = "infeasible_solution"
ERROR_TYPE_SUBOPTIMAL_SOLUTION = "suboptimal_solution"
ERROR_TYPE_FAILED_TO_PARSE = "failed_to_parse"
ERROR_TYPE_EXCEEDED_TOOL_CALL_BUDGET = "exceeded_tool_call_budget"
ERROR_TYPE_TOOL_RETURNED_ERROR = "tool_returned_error"
ERROR_TYPE_WRONG_SHAPE = "wrong_shape"

ERROR_TYPES = frozenset({
    ERROR_TYPE_NONE,
    ERROR_TYPE_INFEASIBLE_SOLUTION,
    ERROR_TYPE_SUBOPTIMAL_SOLUTION,
    ERROR_TYPE_FAILED_TO_PARSE,
    ERROR_TYPE_EXCEEDED_TOOL_CALL_BUDGET,
    ERROR_TYPE_TOOL_RETURNED_ERROR,
    ERROR_TYPE_WRONG_SHAPE,
})


def is_valid_error_type(value: str) -> bool:
    return value in ERROR_TYPES


# ---------------------------------------------------------------------------
# Optimality tolerances
# ---------------------------------------------------------------------------
# Markowitz: relative tolerance on the variance objective.
# A submitted set of weights w_sub is "optimal" if it is feasible AND
#     |variance(w_sub) - optimal_variance| / optimal_variance <= MARKOWITZ_OPTIMALITY_REL_TOL
# Reasoning: CVXPY/CLARABEL converges to ~1e-7 in our reference solves; 1e-4 is comfortably
# above solver noise yet tight enough that even small misallocations trip "suboptimal".
MARKOWITZ_OPTIMALITY_REL_TOL = 1e-4

# TSP-TW: absolute tolerance on the total_time objective, in time units.
# A submitted tour is "optimal" if it is feasible AND
#     |total_time(tour) - optimal_total_time| <= TSP_TW_OPTIMALITY_ABS_TOL
# total_time is integer because travel times are round(euclidean/speed) and service times
# are integer; the reference solver also works in integer time. Strict equality (0) is the
# right default — a model that follows the spec's integer arithmetic gets the same integer
# objective. Bump to 1 if a particular model's MILP formulation uses different rounding and
# we want to be charitable about off-by-one rounding artifacts.
TSP_TW_OPTIMALITY_ABS_TOL = 0


# ---------------------------------------------------------------------------
# Agent-loop budgets (mirrored on the agent provider; kept here so the
# analysis layer can describe them in the writeup without importing the
# provider class)
# ---------------------------------------------------------------------------
AGENT_MAX_ATTEMPTS = 3
AGENT_MAX_TOOL_CALLS_PER_ATTEMPT = 5
