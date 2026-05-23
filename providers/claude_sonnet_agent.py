"""Claude Sonnet agentic provider — multi-turn with tool calls.

Uses the Anthropic SDK tool-use loop. Tools available: solve_qp, solve_milp
(from `tools.solver_tools`).

Attempt model
=============
- Each attempt is its OWN conversation (fresh `messages` list). Carryover
  between attempts is a single user message summarizing what the previous
  attempt produced and why it was rejected. This avoids leaving dangling
  `tool_use` blocks in the history when a turn truncates at max_tokens.
- An attempt ends when the model emits a final answer (stop_reason ==
  "end_turn"), exceeds the per-attempt tool-call budget, hits the per-turn
  output-token limit, or errors out.
- MAX_TOOL_CALLS_PER_ATTEMPT counts individual `tool_use` blocks executed.
  Parallel `tool_use` blocks in one assistant turn count separately.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from anthropic import Anthropic, APIStatusError, RateLimitError

from constants import (
    AGENT_MAX_ATTEMPTS as MAX_ATTEMPTS,
    AGENT_MAX_TOOL_CALLS_PER_ATTEMPT as MAX_TOOL_CALLS_PER_ATTEMPT,
    ERROR_TYPE_EXCEEDED_TOOL_CALL_BUDGET,
    ERROR_TYPE_NONE,
)
from providers.base import AgentToolCall, Provider, ProviderResponse
from scoring import evaluate, feedback_message
from tools.solver_tools import solve_milp, solve_qp


MAX_TOKENS_PER_TURN = 8192


SOLVE_QP_TOOL = {
    "name": "solve_qp",
    "description": (
        "Solve a convex quadratic program. "
        "Minimizes (1/2) x^T P x + q^T x subject to G x <= h, A x == b, and per-coordinate bounds. "
        "P must be symmetric with all eigenvalues >= 0 within numerical tolerance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "P": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "q": {"type": "array", "items": {"type": "number"}},
            "G": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "h": {"type": "array", "items": {"type": "number"}},
            "A": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "b": {"type": "array", "items": {"type": "number"}},
            "bounds": {
                "type": "array",
                "items": {"type": "array", "minItems": 2, "maxItems": 2},
                "description": "Length-n list of [lower, upper] pairs (null = unbounded on that side).",
            },
        },
        "required": ["P", "q"],
    },
}

SOLVE_MILP_TOOL = {
    "name": "solve_milp",
    "description": (
        "Solve a mixed-integer linear program. "
        "Minimizes c^T x subject to A_ub x <= b_ub, A_eq x == b_eq, and per-coordinate bounds. "
        "integrality[i] = 1 forces x[i] integer, 0 leaves it continuous."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "c": {"type": "array", "items": {"type": "number"}},
            "A_ub": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "b_ub": {"type": "array", "items": {"type": "number"}},
            "A_eq": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "b_eq": {"type": "array", "items": {"type": "number"}},
            "bounds": {
                "type": "array",
                "items": {"type": "array", "minItems": 2, "maxItems": 2},
            },
            "integrality": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["c"],
    },
}


def _truncate_args(args: Dict[str, Any], max_chars: int = 240) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def _truncate_result_for_model(result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result)
    if isinstance(out.get("x"), list) and len(out["x"]) > 256:
        out["x"] = out["x"][:256] + ["..."]
    return out


class ClaudeSonnetAgentProvider(Provider):
    name = "claude_sonnet_agent"

    def __init__(self, prices_path: str = "prices.json"):
        self.client = Anthropic()
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        with open(prices_path) as f:
            prices = json.load(f)
        self.price = _resolve_price(prices, self.model)

    def call(
        self,
        prompt: str,
        instance: Dict[str, Any],
        temperature: float,
    ) -> ProviderResponse:
        t_start = time.perf_counter()
        tool_call_log: List[AgentToolCall] = []
        in_tok_total = 0
        out_tok_total = 0
        cost_total = 0.0
        final_text = ""
        final_used_tools = False
        early_err: Optional[str] = None
        attempt = 0
        carryover: Optional[str] = None     # user message seeded at the start of attempts >= 2

        for attempt in range(1, MAX_ATTEMPTS + 1):
            messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
            if carryover:
                messages.append({"role": "user", "content": carryover})

            outcome = self._run_attempt(
                messages=messages,
                attempt=attempt,
                temperature=temperature,
                tool_call_log=tool_call_log,
            )
            in_tok_total += outcome["input_tokens"]
            out_tok_total += outcome["output_tokens"]
            cost_total += outcome["cost_usd"]
            if outcome["text"]:
                final_text = outcome["text"]
            if outcome["used_tools"]:
                final_used_tools = True

            status = outcome["status"]
            if status == "ended":
                ev = evaluate(outcome["text"], instance)
                if ev["error_type"] == ERROR_TYPE_NONE:
                    early_err = None
                    break
                if attempt < MAX_ATTEMPTS:
                    fb = feedback_message(ev) or "Please try again."
                    carryover = (
                        "Your previous attempt produced this answer (excerpt):\n"
                        f"```\n{outcome['text'][:500]}\n```\n{fb}"
                    )
                    early_err = None
                    continue
                early_err = None
                break

            if status == "budget_exceeded":
                early_err = ERROR_TYPE_EXCEEDED_TOOL_CALL_BUDGET
                if attempt < MAX_ATTEMPTS:
                    carryover = (
                        f"Your previous attempt exceeded the {MAX_TOOL_CALLS_PER_ATTEMPT}-tool-call "
                        f"budget without producing a final answer. Try a different approach with "
                        f"fewer tool calls."
                    )
                    continue
                break

            if status == "max_tokens":
                early_err = "tool_returned_error"   # closest existing enum slot; cf. constants.py
                if attempt < MAX_ATTEMPTS:
                    carryover = (
                        "Your previous attempt ran out of output tokens mid-response. "
                        "Be more concise and avoid emitting very large matrices as tool arguments."
                    )
                    continue
                break

            # provider error — don't retry
            early_err = outcome.get("early_error_type") or "provider_error"
            break

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return ProviderResponse(
            raw_text=final_text,
            latency_ms=latency_ms,
            input_tokens=in_tok_total,
            output_tokens=out_tok_total,
            cost_usd=cost_total,
            attempt_number=attempt,
            tool_calls=tool_call_log,
            final_answer_used_tools=final_used_tools,
            early_error_type=early_err,
        )

    def _run_attempt(
        self,
        messages: List[Dict[str, Any]],
        attempt: int,
        temperature: float,
        tool_call_log: List[AgentToolCall],
    ) -> Dict[str, Any]:
        tool_calls_this_attempt = 0
        in_tok = 0
        out_tok = 0
        cost = 0.0
        used_tools = False
        last_text = ""
        turn = 0

        while True:
            turn += 1
            try:
                resp = self._messages_create(messages, temperature)
            except Exception as e:
                return {
                    "status": "error",
                    "text": last_text,
                    "used_tools": used_tools,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                    "early_error_type": f"provider_error: {type(e).__name__}: {e}",
                }
            in_tok += int(resp.usage.input_tokens)
            out_tok += int(resp.usage.output_tokens)
            cost += (
                int(resp.usage.input_tokens) * self.price["input_per_mtok"] / 1_000_000
                + int(resp.usage.output_tokens) * self.price["output_per_mtok"] / 1_000_000
            )

            text_parts: List[str] = []
            tool_use_blocks = []
            for block in resp.content:
                btype = getattr(block, "type", "")
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_use":
                    tool_use_blocks.append(block)
            if text_parts:
                last_text = "".join(text_parts)

            stop_reason = resp.stop_reason

            # If the turn was truncated at max_tokens, ANY tool_use blocks are
            # incomplete and the assistant turn must not be added to history
            # (would leave dangling tool_use ids for the next API call).
            if stop_reason == "max_tokens":
                return {
                    "status": "max_tokens",
                    "text": last_text,
                    "used_tools": used_tools,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": cost,
                }

            messages.append({"role": "assistant", "content": resp.content})

            if stop_reason == "tool_use" and tool_use_blocks:
                if tool_calls_this_attempt + len(tool_use_blocks) > MAX_TOOL_CALLS_PER_ATTEMPT:
                    return {
                        "status": "budget_exceeded",
                        "text": last_text,
                        "used_tools": used_tools,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cost_usd": cost,
                    }
                tool_results = []
                for tu in tool_use_blocks:
                    tool_calls_this_attempt += 1
                    used_tools = True
                    result = self._execute_tool(tu.name, dict(tu.input))
                    tool_call_log.append(AgentToolCall(
                        attempt=attempt,
                        turn=turn,
                        tool=tu.name,
                        args_summary=_truncate_args(dict(tu.input)),
                        status=str(result.get("status", "")),
                        objective=result.get("objective"),
                        message=str(result.get("message", ""))[:240],
                    ))
                    truncated = _truncate_result_for_model(result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(truncated, ensure_ascii=False),
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            # end_turn or any non-tool stop reason: the attempt produced a final response.
            return {
                "status": "ended",
                "text": last_text,
                "used_tools": used_tools,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": cost,
            }

    def _messages_create(self, messages: List[Dict[str, Any]], temperature: float):
        delay = 1.0
        last_err: Optional[Exception] = None
        for _ in range(6):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS_PER_TURN,
                    temperature=temperature,
                    tools=[SOLVE_QP_TOOL, SOLVE_MILP_TOOL],
                    messages=messages,
                )
            except RateLimitError as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 32)
            except APIStatusError as e:
                if e.status_code in (429, 529):
                    last_err = e
                    time.sleep(delay)
                    delay = min(delay * 2, 32)
                else:
                    raise
        raise RuntimeError(f"Claude API: retries exhausted: {last_err}")

    @staticmethod
    def _execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "solve_qp":
            try:
                return solve_qp(**args)
            except TypeError as e:
                return {"status": "error", "x": None, "objective": None,
                        "message": f"bad arguments to solve_qp: {e}"}
        if name == "solve_milp":
            try:
                return solve_milp(**args)
            except TypeError as e:
                return {"status": "error", "x": None, "objective": None,
                        "message": f"bad arguments to solve_milp: {e}"}
        return {"status": "error", "x": None, "objective": None,
                "message": f"unknown tool: {name}"}


def _resolve_price(prices: Dict, model: str) -> Dict[str, float]:
    candidates = [k for k in prices if not k.startswith("_") and model.startswith(k)]
    if not candidates:
        raise ValueError(f"No price entry for model {model}; edit prices.json")
    key = max(candidates, key=len)
    return prices[key]
