# llm-engine-teardowns

Companion repo for the article *"When the LLM recognizes the math, and when it can't count its own variables."*

## The experiment

This benchmarks two classical optimization problems: Markowitz mean-variance
portfolio selection (30 instances, convex QP) and TSP with time windows
(20 instances, MILP). Each problem is solved three ways: a deterministic
solver as ground truth, a raw LLM returning JSON, and an agentic LLM with
access to constrained `solve_qp` and `solve_milp` tools. Two frontier models
on the LLM side: Claude Sonnet 4.6 and Gemini 2.5 Pro. Temperature 0, k=1 for
most cells (the cot agent strategy on Markowitz was run k=3). The run that
produced the checked-in CSV took 444 LLM calls (plus 50 free deterministic
solves) and about $35 in API spend.

## Link to the Medium article

[link to Medium article, to be added on publication]

## What's in this repo

`instances.jsonl` at the root has all 50 problems with their pre-solved ground
truth. `prompts/` has the ten prompt templates (raw zero-shot and CoT for each
problem, plus four agent variants). `solvers/` has the reference CVXPY and
OR-Tools solvers; `tools/solver_tools.py` is the constrained `solve_qp` /
`solve_milp` surface exposed to the agentic provider. `providers/` and
`runner.py` are the experiment harness; `run.py` is the CLI on top.
`results/` holds the pre-baked artifacts from the run that backed the article:
`results.csv`, the accuracy and cost/latency markdown summaries, and a JSONL
of every failure for hand-inspection.

The code is here for inspection and for anyone who wants to re-run a cell or
sweep against their own API keys. The canonical numbers are the ones in
`results/`.

## Dataset size

`results/results.csv` holds **494 evaluated rows**:

- **50** deterministic reference solves (30 Markowitz + 20 TSP-TW)
- **200** raw-LLM rows (100 Sonnet + 100 Gemini, across zero-shot and CoT)
- **244** Sonnet-agent rows

Most cells are k=1; the Markowitz CoT-agent cells were run k=3 (cold 30×3=90,
guided 18×3=54). See [`results/accuracy.md`](results/accuracy.md) for the exact
n per (provider, strategy, problem).

## Running an experiment

Tested with Python 3.14; the pinned `numpy` / `pandas` / `scipy` versions
require Python 3.11+.

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your API keys
python run.py --provider sonnet_raw --strategy zero_shot --problem markowitz
```

Runs write to `out/results.csv` by default, so the canonical `results/`
artifacts are never clobbered. Each cell accepts `--limit`, `--k`,
`--temperature`, `--concurrency`, `--resume`, and `--max-cost-usd`.

A free smoke test (no API key, deterministic solver only):

```bash
python run.py --provider deterministic --strategy reference --problem markowitz --limit 3
python run.py --provider deterministic --strategy reference --problem tsp_tw --limit 3
```

The cheapest LLM check, capped at $1 of spend:

```bash
python run.py --provider sonnet_raw --strategy zero_shot --problem markowitz --limit 1 --max-cost-usd 1
```

## Regenerating the summaries

`results/accuracy.md`, `results/cost_latency.md`, and `results/failures.jsonl`
are all derived from `results/results.csv` by a single script:

```bash
python analyze.py                       # regenerate all three in results/
python analyze.py --out-dir /tmp/check  # write elsewhere, leaving results/ untouched
```

LLM calls are not deterministic in practice. Even at temperature 0, the
provider stack (caching, batching, occasional silent retries on their end)
produces small variations. Re-running will not reproduce the checked-in CSV
exactly. The point of the checked-in numbers is the *aggregate* picture
(roughly two orders of magnitude apart on Markowitz, roughly equivalent on
TSP-TW); that picture has been stable across the smaller re-runs I did while
writing.

## Results summary

On Markowitz the raw LLM produces *feasible* portfolios most of the time
(Sonnet 90% feasible at zero-shot, Gemini 83%) but the variance is on average
146% (Sonnet) and 170% (Gemini) above optimal. Every raw call is suboptimal
by enough that you would never deploy it. The agentic Sonnet, with the
`solve_qp` tool, hits 100% optimal across all four agent strategies.

On TSP-TW the situation inverts. The raw LLM is surprisingly decent: a ~1-2%
mean optimality gap on feasible tours, with Sonnet feasible ~90-95% of the time
and Gemini more variable (45-80%). The agentic Sonnet, with the `solve_milp`
tool, is feasible on zero of twenty instances in both agent strategies tested
(0/20 cold, 0/20 guided). It fails to write down a correct MILP constraint
matrix, hits the tool's shape-validation errors, and gives up. The article
walks through exactly where the MTZ formulation breaks under the model's
bookkeeping.

Full per-strategy breakdown is in [`results/accuracy.md`](results/accuracy.md);
cost and latency in [`results/cost_latency.md`](results/cost_latency.md); a
sample of failure transcripts in [`results/failures.jsonl`](results/failures.jsonl).
Each line of that JSONL is one failed run with its full context: the instance,
the prompt strategy, the `error_type`, the model's raw response and parsed
answer, every tool call it made, and the feasibility message explaining why the
answer was rejected.

## A note on cost

Re-running the entire sweep costs roughly $30 in Anthropic API calls and ~$4
in Gemini calls — about $35 total; see
[`results/cost_latency.md`](results/cost_latency.md) for the exact per-provider
breakdown. The deterministic provider is free. If you want to reproduce
a slice, the cheapest path is the deterministic solver against the stored
instances; the most expensive single cell is the Sonnet agent on TSP-TW,
where each attempt can cost on the order of fifty cents because the model
emits large matrices as tool arguments and the tool refuses them and the
model retries.

## Honest limitations

Temperature was held at 0 on all runs; a variance sweep at higher
temperatures was planned but not completed before the article shipped. Two
optimization problems is not a complete picture of the optimization
landscape. Markowitz is a small dense QP, TSP-TW is a small but combinatorial
MILP, and the failure modes for, say, large-scale stochastic programs or
non-convex MINLPs would look different. Only Sonnet 4.6 and Gemini 2.5 Pro
were tested; Opus 4.7 and Gemini 3.1 Pro might well solve the TSP-TW agent
case. The TSP-TW agent failures are all on the textbook MTZ formulation; I
did not try alternative formulations (DFJ subtour-elimination, lazy
constraint loops) that might survive the bookkeeping better.

One wording note on the prompts: the guided TSP-TW agent prompt
(`prompts/tsp_tw_zero_shot_agent_guided.txt`) describes the objective as travel
+ service time "or equivalently the return arrival time at the depot." Those are
*not* equivalent once a tour has to wait for a time window to open — the
reference solver minimizes elapsed/return-arrival time *including* waiting (see
the objective note in `solvers/tsp_tw_ortools.py`). The prompt is kept verbatim
because it is the exact text that produced the checked-in results, and every
TSP-TW agent run errored on MILP shape validation before ever calling the
solver, so the imperfect objective wording had no effect on the outcome.

If you find a bug or want to push back on a result, an issue or a PR is welcome.

## License and attribution

MIT, see [`LICENSE`](LICENSE). Written by Alex Spivakovsky in 2026.
