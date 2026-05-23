# Accuracy

Cells are aggregated over k samples and instances within each (provider, strategy, problem).
`feasibility_rate` = fraction of runs whose answer satisfied all problem constraints.
`optimal_rate` = fraction within the optimality tolerance defined in `constants.py`
(Markowitz: rel. variance gap ≤ 1e-4; TSP-TW: total_time exactly equal).
Optimality-gap statistics are computed over feasible runs only.
Wilson 95% confidence intervals shown for both rates.

## Feasibility and optimality rates

| problem   | provider            | strategy               |   n |   feasible |   feas_ci_lo |   feas_ci_hi |   optimal |   opt_ci_lo |   opt_ci_hi |
|:----------|:--------------------|:-----------------------|----:|-----------:|-------------:|-------------:|----------:|------------:|------------:|
| markowitz | claude_sonnet_agent | cot_agent_cold         |  90 |      1.000 |        0.959 |        1.000 |     1.000 |       0.959 |       1.000 |
| markowitz | claude_sonnet_agent | cot_agent_guided       |  54 |      1.000 |        0.934 |        1.000 |     1.000 |       0.934 |       1.000 |
| markowitz | claude_sonnet_agent | zero_shot_agent_cold   |  30 |      1.000 |        0.886 |        1.000 |     1.000 |       0.886 |       1.000 |
| markowitz | claude_sonnet_agent | zero_shot_agent_guided |  30 |      1.000 |        0.886 |        1.000 |     1.000 |       0.886 |       1.000 |
| markowitz | claude_sonnet_raw   | cot_raw                |  30 |      0.767 |        0.591 |        0.882 |     0.000 |       0.000 |       0.114 |
| markowitz | claude_sonnet_raw   | zero_shot_raw          |  30 |      0.900 |        0.744 |        0.965 |     0.000 |       0.000 |       0.114 |
| markowitz | deterministic_opt   | reference              |  30 |      1.000 |        0.886 |        1.000 |     1.000 |       0.886 |       1.000 |
| markowitz | gemini_raw          | cot_raw                |  30 |      0.867 |        0.703 |        0.947 |     0.000 |       0.000 |       0.114 |
| markowitz | gemini_raw          | zero_shot_raw          |  30 |      0.833 |        0.664 |        0.927 |     0.000 |       0.000 |       0.114 |
| tsp_tw    | claude_sonnet_agent | zero_shot_agent_cold   |  20 |      0.000 |        0.000 |        0.161 |     0.000 |       0.000 |       0.161 |
| tsp_tw    | claude_sonnet_agent | zero_shot_agent_guided |  20 |      0.000 |        0.000 |        0.161 |     0.000 |       0.000 |       0.161 |
| tsp_tw    | claude_sonnet_raw   | cot_raw                |  20 |      0.950 |        0.764 |        0.991 |     0.150 |       0.052 |       0.360 |
| tsp_tw    | claude_sonnet_raw   | zero_shot_raw          |  20 |      0.900 |        0.699 |        0.972 |     0.200 |       0.081 |       0.416 |
| tsp_tw    | deterministic_opt   | reference              |  20 |      1.000 |        0.839 |        1.000 |     1.000 |       0.839 |       1.000 |
| tsp_tw    | gemini_raw          | cot_raw                |  20 |      0.800 |        0.584 |        0.919 |     0.300 |       0.145 |       0.519 |
| tsp_tw    | gemini_raw          | zero_shot_raw          |  20 |      0.450 |        0.258 |        0.658 |     0.100 |       0.028 |       0.301 |

## Optimality gap and error-mode breakdown

Gap statistics computed over **feasible** rows only. Error-mode columns count rows by `error_type`.

| problem   | provider            | strategy               |   n |   mean_gap |   p50_gap |   p95_gap |   suboptimal |   infeasible |   unparseable |   wrong_shape |   exceeded_budget |   tool_err |
|:----------|:--------------------|:-----------------------|----:|-----------:|----------:|----------:|-------------:|-------------:|--------------:|--------------:|------------------:|-----------:|
| markowitz | claude_sonnet_agent | cot_agent_cold         |  90 |     0.0000 |    0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| markowitz | claude_sonnet_agent | cot_agent_guided       |  54 |     0.0000 |   -0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| markowitz | claude_sonnet_agent | zero_shot_agent_cold   |  30 |     0.0000 |    0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| markowitz | claude_sonnet_agent | zero_shot_agent_guided |  30 |    -0.0000 |   -0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| markowitz | claude_sonnet_raw   | cot_raw                |  30 |     0.8645 |    0.5708 |    2.2986 |           23 |            4 |             3 |             0 |                 0 |          0 |
| markowitz | claude_sonnet_raw   | zero_shot_raw          |  30 |     1.4612 |    0.9212 |    4.7914 |           27 |            3 |             0 |             0 |                 0 |          0 |
| markowitz | deterministic_opt   | reference              |  30 |     0.0000 |    0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| markowitz | gemini_raw          | cot_raw                |  30 |     1.0845 |    0.9919 |    2.3266 |           26 |            3 |             1 |             0 |                 0 |          0 |
| markowitz | gemini_raw          | zero_shot_raw          |  30 |     1.7045 |    1.4690 |    4.1308 |           25 |            5 |             0 |             0 |                 0 |          0 |
| tsp_tw    | claude_sonnet_agent | zero_shot_agent_cold   |  20 |   nan      |  nan      |  nan      |            0 |            0 |             0 |             0 |                 0 |         20 |
| tsp_tw    | claude_sonnet_agent | zero_shot_agent_guided |  20 |   nan      |  nan      |  nan      |            0 |            0 |             0 |             0 |                 0 |         15 |
| tsp_tw    | claude_sonnet_raw   | cot_raw                |  20 |     0.0113 |    0.0088 |    0.0539 |           16 |            0 |             1 |             0 |                 0 |          0 |
| tsp_tw    | claude_sonnet_raw   | zero_shot_raw          |  20 |     0.0109 |    0.0130 |    0.0639 |           14 |            2 |             0 |             0 |                 0 |          0 |
| tsp_tw    | deterministic_opt   | reference              |  20 |     0.0000 |    0.0000 |    0.0000 |            0 |            0 |             0 |             0 |                 0 |          0 |
| tsp_tw    | gemini_raw          | cot_raw                |  20 |     0.0015 |    0.0043 |    0.0197 |           10 |            0 |             4 |             0 |                 0 |          0 |
| tsp_tw    | gemini_raw          | zero_shot_raw          |  20 |    -0.0031 |    0.0086 |    0.0176 |            7 |            8 |             3 |             0 |                 0 |          0 |

