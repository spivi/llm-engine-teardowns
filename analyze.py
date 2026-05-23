"""Produce `accuracy.md`, `cost_latency.md`, and a failures JSONL from a results CSV.

By default reads `results/results.csv` and writes the three artifacts back into
`results/`. Pass --out-dir to write elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    return str(v).strip().lower() in ("true", "1", "yes")


def _maybe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        if not np.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["feasible_b"] = df["feasible"].apply(_coerce_bool)
    df["optimality_gap_f"] = df["optimality_gap"].apply(_maybe_float)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0.0)
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).astype(int)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)
    return df


def load_instances(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            out[inst["instance_id"]] = inst
    return out


def write_md(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content + "\n")


def build_accuracy_table(df: pd.DataFrame) -> str:
    rows = []
    for (prov, strat, sub), g in df.groupby(["provider", "strategy", "subproblem"]):
        n = len(g)
        feasible = int(g["feasible_b"].sum())
        optimal_within_tol = int((g["error_type"] == "none").sum())
        suboptimal = int((g["error_type"] == "suboptimal_solution").sum())
        infeasible = int((g["error_type"] == "infeasible_solution").sum())
        unparseable = int((g["error_type"] == "failed_to_parse").sum())
        wrong_shape = int((g["error_type"] == "wrong_shape").sum())
        exceeded_budget = int((g["error_type"] == "exceeded_tool_call_budget").sum())
        tool_err = int((g["error_type"] == "tool_returned_error").sum())
        gaps = g.loc[g["feasible_b"], "optimality_gap_f"].dropna()
        mean_gap = float(gaps.mean()) if len(gaps) else float("nan")
        p50_gap = float(gaps.median()) if len(gaps) else float("nan")
        p95_gap = float(gaps.quantile(0.95)) if len(gaps) else float("nan")
        f_lo, f_hi = _wilson(feasible, n)
        o_lo, o_hi = _wilson(optimal_within_tol, n)
        rows.append({
            "provider": prov, "strategy": strat, "problem": sub,
            "n": n,
            "feasibility_rate": feasible / n if n else 0.0,
            "feas_ci_lo": f_lo, "feas_ci_hi": f_hi,
            "optimal_rate": optimal_within_tol / n if n else 0.0,
            "opt_ci_lo": o_lo, "opt_ci_hi": o_hi,
            "mean_gap": mean_gap, "p50_gap": p50_gap, "p95_gap": p95_gap,
            "suboptimal": suboptimal,
            "infeasible": infeasible,
            "unparseable": unparseable,
            "wrong_shape": wrong_shape,
            "exceeded_budget": exceeded_budget,
            "tool_err": tool_err,
        })
    table = pd.DataFrame(rows).sort_values(["problem", "provider", "strategy"]).reset_index(drop=True)

    lines = []
    lines.append("# Accuracy")
    lines.append("")
    lines.append("Cells are aggregated over k samples and instances within each (provider, strategy, problem).")
    lines.append("`feasibility_rate` = fraction of runs whose answer satisfied all problem constraints.")
    lines.append("`optimal_rate` = fraction within the optimality tolerance defined in `constants.py`")
    lines.append("(Markowitz: rel. variance gap ≤ 1e-4; TSP-TW: total_time exactly equal).")
    lines.append("Optimality-gap statistics are computed over feasible runs only.")
    lines.append("Wilson 95% confidence intervals shown for both rates.")
    lines.append("")

    display_a = table[[
        "problem", "provider", "strategy", "n",
        "feasibility_rate", "feas_ci_lo", "feas_ci_hi",
        "optimal_rate", "opt_ci_lo", "opt_ci_hi",
    ]].rename(columns={
        "feasibility_rate": "feasible",
        "optimal_rate": "optimal",
    })
    lines.append("## Feasibility and optimality rates")
    lines.append("")
    lines.append(display_a.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    display_b = table[[
        "problem", "provider", "strategy", "n",
        "mean_gap", "p50_gap", "p95_gap",
        "suboptimal", "infeasible", "unparseable", "wrong_shape",
        "exceeded_budget", "tool_err",
    ]]
    lines.append("## Optimality gap and error-mode breakdown")
    lines.append("")
    lines.append("Gap statistics computed over **feasible** rows only. Error-mode columns count rows by `error_type`.")
    lines.append("")
    lines.append(display_b.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return "\n".join(lines)


def build_cost_latency(df: pd.DataFrame) -> str:
    rows = []
    for (prov, strat, sub), g in df.groupby(["provider", "strategy", "subproblem"]):
        n = len(g)
        rows.append({
            "provider": prov, "strategy": strat, "problem": sub,
            "n": n,
            "total_cost_usd": float(g["cost_usd"].sum()),
            "mean_cost_usd": float(g["cost_usd"].mean()) if n else 0.0,
            "p95_cost_usd": float(g["cost_usd"].quantile(0.95)) if n else 0.0,
            "mean_latency_s": float(g["latency_ms"].mean() / 1000) if n else 0.0,
            "p50_latency_s": float(g["latency_ms"].median() / 1000) if n else 0.0,
            "p95_latency_s": float(g["latency_ms"].quantile(0.95) / 1000) if n else 0.0,
            "mean_input_tokens": float(g["input_tokens"].mean()) if n else 0.0,
            "mean_output_tokens": float(g["output_tokens"].mean()) if n else 0.0,
        })
    by_cell = pd.DataFrame(rows).sort_values(["problem", "provider", "strategy"])

    prov_rows = []
    for prov, g in df.groupby("provider"):
        prov_rows.append({
            "provider": prov,
            "n_calls": len(g),
            "total_cost_usd": float(g["cost_usd"].sum()),
            "mean_cost_per_call": float(g["cost_usd"].mean()),
            "total_input_tokens": int(g["input_tokens"].sum()),
            "total_output_tokens": int(g["output_tokens"].sum()),
            "mean_latency_s": float(g["latency_ms"].mean() / 1000),
            "p95_latency_s": float(g["latency_ms"].quantile(0.95) / 1000),
        })
    by_prov = pd.DataFrame(prov_rows).sort_values("provider")
    grand_total = float(df["cost_usd"].sum())

    lines = []
    lines.append("# Cost & latency")
    lines.append("")
    lines.append(f"**Total spend: ${grand_total:.3f}**")
    lines.append("")
    lines.append("## Per provider")
    lines.append("")
    lines.append(by_prov.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Per (provider, strategy, problem)")
    lines.append("")
    lines.append(by_cell.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return "\n".join(lines)


def pick_failures(
    df: pd.DataFrame,
    instances: Dict[str, Dict[str, Any]],
    per_problem: int,
) -> List[Dict[str, Any]]:
    """Pick diverse failure rows per problem (max `per_problem`).

    Diversity: prefer different (provider, strategy, error_type) combinations.
    A failure is any row with error_type != "none".
    """
    picked: List[Dict[str, Any]] = []
    for problem in sorted(df["subproblem"].unique()):
        prob_df = df[(df["subproblem"] == problem) & (df["error_type"] != "none")]
        if prob_df.empty:
            continue
        seen_keys = set()
        chosen_rows: List[pd.Series] = []
        for _, row in prob_df.iterrows():
            key = (row["provider"], row["strategy"], row["error_type"])
            if key in seen_keys:
                continue
            chosen_rows.append(row)
            seen_keys.add(key)
            if len(chosen_rows) >= per_problem:
                break
        if len(chosen_rows) < per_problem:
            picked_ids = {id(r) for r in chosen_rows}
            for _, row in prob_df.iterrows():
                if id(row) in picked_ids:
                    continue
                chosen_rows.append(row)
                if len(chosen_rows) >= per_problem:
                    break
        for row in chosen_rows:
            iid = row["instance_id"]
            picked.append({
                "problem": problem,
                "instance_id": iid,
                "provider": row["provider"],
                "strategy": row["strategy"],
                "error_type": row["error_type"],
                "early_error_type": row.get("early_error_type", "") or "",
                "feasible": bool(row["feasible_b"]),
                "feasibility_message": str(row.get("feasibility_message", "") or ""),
                "optimality_gap": row.get("optimality_gap_f"),
                "model_objective": _maybe_float(row.get("model_objective")),
                "gt_objective": _maybe_float(row.get("gt_objective")),
                "attempt_number": _maybe_float(row.get("attempt_number")),
                "n_tool_calls": int(row.get("n_tool_calls", 0) or 0),
                "tool_calls": _safe_json(row.get("tool_calls"), []),
                "raw_response": row.get("raw_response", ""),
                "parsed_answer": _safe_json(row.get("parsed_answer"), None),
                "cost_usd": float(row.get("cost_usd", 0.0) or 0.0),
                "latency_s": float(row.get("latency_ms", 0.0) or 0.0) / 1000,
                "instance": instances.get(iid, {}),
            })
    return picked


def _safe_json(s, default):
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--instances", default="instances.jsonl")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--failures-per-problem", type=int, default=5)
    args = parser.parse_args()

    df = load_results(args.results)
    instances = load_instances(args.instances)

    write_md(os.path.join(args.out_dir, "accuracy.md"), build_accuracy_table(df))
    write_md(os.path.join(args.out_dir, "cost_latency.md"), build_cost_latency(df))

    picked = pick_failures(df, instances, args.failures_per_problem)
    failures_jsonl = os.path.join(args.out_dir, "failures.jsonl")
    with open(failures_jsonl, "w") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Wrote:\n  {args.out_dir}/accuracy.md")
    print(f"  {args.out_dir}/cost_latency.md")
    print(f"  {failures_jsonl}")


if __name__ == "__main__":
    main()
