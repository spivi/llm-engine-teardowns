"""Deterministic optimization provider: wraps the reference solvers.

Dispatches by subproblem:
    markowitz -> solvers.markowitz_cvxpy.solve
    tsp_tw    -> solvers.tsp_tw_ortools.solve

The "raw_text" returned mirrors what the LLM would return (a JSON answer) so the
runner's scoring pipeline can treat all providers uniformly.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from providers.base import Provider, ProviderResponse
from solvers import markowitz_cvxpy, tsp_tw_ortools


class DeterministicProvider(Provider):
    name = "deterministic"

    def call(
        self,
        prompt: str,
        instance: Dict[str, Any],
        temperature: float,
    ) -> ProviderResponse:
        t0 = time.perf_counter()
        sub = instance["subproblem"]
        if sub == "markowitz":
            res = markowitz_cvxpy.solve(
                instance["returns"], instance["cov_matrix"], instance["target_return"]
            )
            raw = json.dumps({"weights": res["weights"]}, ensure_ascii=False)
        elif sub == "tsp_tw":
            res = tsp_tw_ortools.solve(
                instance["coordinates"], instance["service_time"],
                instance["time_windows"], speed=instance["speed"],
            )
            raw = json.dumps({"tour": res["tour"]}, ensure_ascii=False)
        else:
            raise ValueError(f"unknown subproblem: {sub!r}")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ProviderResponse(
            raw_text=raw,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
