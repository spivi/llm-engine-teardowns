"""Experiment runner: iterates (provider × strategy × instance × k), writes append-only CSV.

Each provider declares which strategy labels it supports (the labels match prompt
filenames under `--prompts-dir`). The runner skips incompatible (provider, strategy)
pairs silently so a single comma-separated --strategies list can be passed for any
provider mix.

Scoring happens here, via `scoring.evaluate`, so analysis is cheap and uniform.
"""
from __future__ import annotations

import csv
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from providers.base import Provider, ProviderResponse
from scoring import evaluate


PROVIDER_STRATEGIES: Dict[str, List[str]] = {
    "deterministic": ["reference"],
    "claude_sonnet_raw": ["zero_shot_raw", "cot_raw"],
    "gemini_raw": ["zero_shot_raw", "cot_raw"],
    "claude_sonnet_agent": [
        "zero_shot_agent_guided", "cot_agent_guided",
        "zero_shot_agent_cold", "cot_agent_cold",
    ],
}


CSV_FIELDS = [
    "run_id", "timestamp",
    "instance_id", "subproblem",
    "provider", "strategy",
    "k", "temperature",
    "raw_response",
    "parsed_answer",
    "error_type",
    "feasible",
    "optimality_gap",
    "model_objective",
    "gt_objective",
    "feasibility_message",
    "attempt_number",
    "n_tool_calls",
    "tool_calls",
    "final_answer_used_tools",
    "latency_ms",
    "input_tokens", "output_tokens", "cost_usd",
    "early_error_type",
]


_PROMPT_PLACEHOLDERS = (
    "returns", "cov_matrix", "target_return",
    "coordinates", "time_windows", "service_time", "speed",
)


def load_instances(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_template(prompts_dir: str, subproblem: str, strategy: str) -> str:
    path = os.path.join(prompts_dir, f"{subproblem}_{strategy}.txt")
    with open(path) as f:
        return f.read()


def format_prompt(template: str, instance: Dict[str, Any]) -> str:
    """Targeted placeholder substitution; safe against literal `{...}` in the template."""
    out = template
    for key in _PROMPT_PLACEHOLDERS:
        if key in instance:
            value = instance[key]
            if isinstance(value, (list, dict)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            out = out.replace("{" + key + "}", rendered)
    return out


def build_provider(name: str) -> Provider:
    if name == "deterministic":
        from providers.deterministic import DeterministicProvider
        return DeterministicProvider()
    if name == "claude_sonnet_raw":
        from providers.claude_sonnet_raw import ClaudeSonnetRawProvider
        return ClaudeSonnetRawProvider()
    if name == "claude_sonnet_agent":
        from providers.claude_sonnet_agent import ClaudeSonnetAgentProvider
        return ClaudeSonnetAgentProvider()
    if name == "gemini_raw":
        from providers.gemini_raw import GeminiRawProvider
        return GeminiRawProvider()
    raise ValueError(f"unknown provider: {name}")


def load_completed(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    done = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((
                row["instance_id"], row["provider"], row["strategy"],
                int(row["k"]), float(row["temperature"]),
            ))
    return done


def _row_for(
    inst: Dict[str, Any],
    provider: Provider,
    strategy: str,
    k_idx: int,
    temperature: float,
    resp: Optional[ProviderResponse],
    err: Optional[str],
) -> Dict[str, Any]:
    """Build the CSV row from a provider response (or an exception)."""
    if resp is None:
        return {
            "run_id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "instance_id": inst["instance_id"],
            "subproblem": inst["subproblem"],
            "provider": provider.name,
            "strategy": strategy,
            "k": k_idx,
            "temperature": temperature,
            "raw_response": f"PROVIDER_ERROR: {err}",
            "parsed_answer": "",
            "error_type": "provider_error",
            "feasible": False,
            "optimality_gap": "",
            "model_objective": "",
            "gt_objective": _gt_objective(inst),
            "feasibility_message": "",
            "attempt_number": "",
            "n_tool_calls": 0,
            "tool_calls": "[]",
            "final_answer_used_tools": False,
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "early_error_type": "provider_error",
        }

    ev = evaluate(resp.raw_text, inst)
    error_type = resp.early_error_type or ev["error_type"]
    model_obj = ev.get("variance") if inst["subproblem"] == "markowitz" else ev.get("total_time")

    tool_calls_serialized = json.dumps(
        [asdict(tc) for tc in resp.tool_calls],
        ensure_ascii=False,
    )

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "instance_id": inst["instance_id"],
        "subproblem": inst["subproblem"],
        "provider": provider.name,
        "strategy": strategy,
        "k": k_idx,
        "temperature": temperature,
        "raw_response": resp.raw_text,
        "parsed_answer": json.dumps(ev.get("parsed_answer")) if ev.get("parsed_answer") is not None else "",
        "error_type": error_type,
        "feasible": bool(ev.get("feasible", False)),
        "optimality_gap": "" if ev.get("optimality_gap") is None else float(ev["optimality_gap"]),
        "model_objective": "" if model_obj is None else float(model_obj),
        "gt_objective": _gt_objective(inst),
        "feasibility_message": ev.get("feasibility_message", ""),
        "attempt_number": resp.attempt_number,
        "n_tool_calls": len(resp.tool_calls),
        "tool_calls": tool_calls_serialized,
        "final_answer_used_tools": bool(resp.final_answer_used_tools),
        "latency_ms": round(resp.latency_ms, 2),
        "input_tokens": int(resp.input_tokens),
        "output_tokens": int(resp.output_tokens),
        "cost_usd": round(resp.cost_usd, 6),
        "early_error_type": resp.early_error_type or "",
    }


def _gt_objective(inst: Dict[str, Any]) -> float:
    if inst["subproblem"] == "markowitz":
        return float(inst["optimal_variance"])
    return float(inst["optimal_total_time"])


def run_cell(
    provider: Provider,
    strategy: str,
    inst: Dict[str, Any],
    k_idx: int,
    temperature: float,
    prompts_dir: str,
) -> Dict[str, Any]:
    """Execute one cell. Caller writes the resulting row to CSV."""
    if strategy == "reference":
        prompt = ""
    else:
        template = load_template(prompts_dir, inst["subproblem"], strategy)
        prompt = format_prompt(template, inst)
    try:
        resp = provider.call(prompt, inst, temperature)
        return _row_for(inst, provider, strategy, k_idx, temperature, resp, None)
    except Exception as e:
        return _row_for(inst, provider, strategy, k_idx, temperature, None, f"{type(e).__name__}: {e}")


def run_sweep(
    providers: List[Provider],
    strategies: List[str],
    instances: List[Dict[str, Any]],
    *,
    k: int,
    temperature: float,
    output: str,
    prompts_dir: str,
    concurrency: int = 1,
    resume: bool = False,
    limit: int = 0,
    max_cost_usd: float = 0.0,
    subproblem: Optional[str] = None,
) -> None:
    """Run every compatible (provider, strategy, instance, k) cell, append to `output`."""
    completed = load_completed(output) if resume else set()

    insts = instances if subproblem is None else [i for i in instances if i["subproblem"] == subproblem]

    cells: List[Tuple[Provider, str, Dict[str, Any], int]] = []
    for prov in providers:
        valid = set(PROVIDER_STRATEGIES.get(prov.name, []))
        prov_strategies = [s for s in strategies if s in valid]
        if not prov_strategies and prov.name == "deterministic":
            prov_strategies = ["reference"]
        for s in prov_strategies:
            insts_to_run = insts if limit <= 0 else insts[: limit]
            for inst in insts_to_run:
                for k_idx in range(k):
                    key = (inst["instance_id"], prov.name, s, k_idx, temperature)
                    if key in completed:
                        continue
                    cells.append((prov, s, inst, k_idx))

    if not cells:
        print("Nothing to do (all cells completed or no compatible (provider, strategy) pairs).")
        return

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    write_header = not os.path.exists(output)
    fh = open(output, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
        fh.flush()

    cumulative_cost = 0.0
    print(f"Running {len(cells)} cells (concurrency={concurrency})")

    def _execute(cell):
        prov, s, inst, k_idx = cell
        return run_cell(prov, s, inst, k_idx, temperature, prompts_dir)

    rows_written = 0
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_execute, c): c for c in cells}
            for fut in as_completed(futs):
                row = fut.result()
                writer.writerow(row)
                fh.flush()
                cumulative_cost += float(row.get("cost_usd") or 0.0)
                rows_written += 1
                if rows_written % 10 == 0 or rows_written == len(cells):
                    print(f"  [{rows_written}/{len(cells)}] cum_cost=${cumulative_cost:.3f}")
                if max_cost_usd > 0 and cumulative_cost > max_cost_usd:
                    print(f"!! cumulative cost ${cumulative_cost:.2f} > cap ${max_cost_usd}; aborting")
                    for f in futs:
                        f.cancel()
                    break
    else:
        for i, cell in enumerate(cells, 1):
            row = _execute(cell)
            writer.writerow(row)
            fh.flush()
            cumulative_cost += float(row.get("cost_usd") or 0.0)
            rows_written += 1
            if i % 5 == 0 or i == len(cells):
                print(f"  [{i}/{len(cells)}] cum_cost=${cumulative_cost:.3f}")
            if max_cost_usd > 0 and cumulative_cost > max_cost_usd:
                print(f"!! cumulative cost ${cumulative_cost:.2f} > cap ${max_cost_usd}; aborting")
                break

    fh.close()
    print(f"Wrote {rows_written} rows to {output}. Cumulative cost: ${cumulative_cost:.3f}")
