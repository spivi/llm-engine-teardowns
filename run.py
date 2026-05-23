"""CLI entry point for running one (provider, strategy, problem) slice.

    python run.py --provider sonnet_raw --strategy zero_shot --problem markowitz

Friendly strategy names map to underlying prompt-file labels:
    raw providers:    zero_shot     -> zero_shot_raw
                      cot           -> cot_raw
    sonnet_agent:     agent_cold    -> zero_shot_agent_cold
                      agent_guided  -> zero_shot_agent_guided
    deterministic:    (any)         -> reference

The underlying labels (e.g. `cot_agent_cold` for Markowitz) can be passed directly
to --strategy if needed.

By default writes to `out/results.csv` rather than overwriting `results/results.csv`,
since that file is the canonical artifact backing the article.
"""
from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from runner import PROVIDER_STRATEGIES, build_provider, load_instances, run_sweep


PROVIDER_ALIAS = {
    "sonnet_raw": "claude_sonnet_raw",
    "sonnet_agent": "claude_sonnet_agent",
    "gemini_raw": "gemini_raw",
    "deterministic": "deterministic",
}


STRATEGY_ALIAS = {
    ("claude_sonnet_raw", "zero_shot"): "zero_shot_raw",
    ("claude_sonnet_raw", "cot"): "cot_raw",
    ("gemini_raw", "zero_shot"): "zero_shot_raw",
    ("gemini_raw", "cot"): "cot_raw",
    ("claude_sonnet_agent", "agent_cold"): "zero_shot_agent_cold",
    ("claude_sonnet_agent", "agent_guided"): "zero_shot_agent_guided",
}


def resolve_strategy(provider_name: str, strategy: str) -> str:
    if provider_name == "deterministic":
        return "reference"
    key = (provider_name, strategy)
    if key in STRATEGY_ALIAS:
        return STRATEGY_ALIAS[key]
    valid = PROVIDER_STRATEGIES.get(provider_name, [])
    if strategy in valid:
        return strategy
    raise SystemExit(
        f"strategy {strategy!r} not valid for provider {provider_name!r}. "
        f"Friendly names: {sorted({s for (p, s) in STRATEGY_ALIAS if p == provider_name})}; "
        f"or pass an underlying label directly: {valid}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", required=True, choices=list(PROVIDER_ALIAS))
    parser.add_argument(
        "--strategy", required=True,
        help="zero_shot | cot | agent_cold | agent_guided (or an underlying label).",
    )
    parser.add_argument("--problem", required=True, choices=["markowitz", "tsp_tw"])
    parser.add_argument("--instances", default="instances.jsonl")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--out", default="out/results.csv",
                        help="Output CSV. Defaults to out/results.csv so the canonical results/ artifacts are not clobbered.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0,
                        help="Run only the first N instances. 0 = all.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.0,
                        help="Abort the sweep when cumulative cost exceeds this value. 0 = no cap.")
    args = parser.parse_args()

    load_dotenv()

    provider_internal = PROVIDER_ALIAS[args.provider]
    strategy_label = resolve_strategy(provider_internal, args.strategy)

    try:
        provider = build_provider(provider_internal)
    except Exception as e:
        print(f"error initializing provider {provider_internal}: {e}", file=sys.stderr)
        sys.exit(1)

    instances = load_instances(args.instances)

    run_sweep(
        providers=[provider],
        strategies=[strategy_label],
        instances=instances,
        k=args.k,
        temperature=args.temperature,
        output=args.out,
        prompts_dir=args.prompts_dir,
        concurrency=args.concurrency,
        resume=args.resume,
        limit=args.limit,
        max_cost_usd=args.max_cost_usd,
        subproblem=args.problem,
    )


if __name__ == "__main__":
    main()
