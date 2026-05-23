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
produced the checked-in CSV took roughly 500 LLM calls and about $40 in API
spend.

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

## Running an experiment

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env with your API keys
python run.py --provider sonnet_raw --strategy zero_shot --problem markowitz
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

On TSP-TW the situation inverts. The raw LLM is surprisingly decent: ~1% mean
optimality gap and ~90% feasibility on both models. The agentic Sonnet, with
the `solve_milp` tool, is feasible on zero of twenty instances. It fails to
write down a correct MILP constraint matrix, hits the tool's shape-validation
errors, and gives up. The article walks through exactly where the MTZ
formulation breaks under the model's bookkeeping.

Full per-strategy breakdown is in [`results/accuracy.md`](results/accuracy.md);
cost and latency in [`results/cost_latency.md`](results/cost_latency.md); a
sample of failure transcripts in [`results/failures.jsonl`](results/failures.jsonl).

## A note on cost

Re-running the entire sweep costs roughly $40 in Anthropic API calls and ~$5
in Gemini calls. The deterministic provider is free. If you want to reproduce
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
constraint loops) that might survive the bookkeeping better. If you find a
bug or want to push back on a result, an issue or a PR is welcome.

## License and attribution

MIT, see [`LICENSE`](LICENSE). Written by Alex Spivakovsky in 2026.
