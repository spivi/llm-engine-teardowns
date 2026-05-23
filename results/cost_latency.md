# Cost & latency

**Total spend: $34.504**

## Per provider

| provider            |   n_calls |   total_cost_usd |   mean_cost_per_call |   total_input_tokens |   total_output_tokens |   mean_latency_s |   p95_latency_s |
|:--------------------|----------:|-----------------:|---------------------:|---------------------:|----------------------:|-----------------:|----------------:|
| claude_sonnet_agent |       244 |          28.0665 |               0.1150 |              2163455 |               1438410 |          70.6310 |        277.6939 |
| claude_sonnet_raw   |       100 |           2.4461 |               0.0245 |               104136 |                142248 |          21.9461 |         29.6932 |
| deterministic_opt   |        50 |           0.0000 |               0.0000 |                    0 |                     0 |           0.1285 |          0.5080 |
| gemini_raw          |       100 |           3.9918 |               0.0399 |               191676 |                375219 |          26.8244 |         54.5027 |

## Per (provider, strategy, problem)

| provider            | strategy               | problem   |   n |   total_cost_usd |   mean_cost_usd |   p95_cost_usd |   mean_latency_s |   p50_latency_s |   p95_latency_s |   mean_input_tokens |   mean_output_tokens |
|:--------------------|:-----------------------|:----------|----:|-----------------:|----------------:|---------------:|-----------------:|----------------:|----------------:|--------------------:|---------------------:|
| claude_sonnet_agent | cot_agent_cold         | markowitz |  90 |           5.1263 |          0.0570 |         0.0787 |          45.8117 |         28.7613 |         47.8714 |           7684.8778 |            2260.3000 |
| claude_sonnet_agent | cot_agent_guided       | markowitz |  54 |           2.8161 |          0.0522 |         0.0531 |          37.5650 |         36.0788 |         42.7036 |           7595.1852 |            1957.6852 |
| claude_sonnet_agent | zero_shot_agent_cold   | markowitz |  30 |           1.3968 |          0.0466 |         0.0497 |          23.7889 |         17.7982 |         36.2555 |           7056.2667 |            1692.8333 |
| claude_sonnet_agent | zero_shot_agent_guided | markowitz |  30 |           1.3629 |          0.0454 |         0.0459 |          31.4143 |         30.1555 |         34.8059 |           7146.2667 |            1599.4333 |
| claude_sonnet_raw   | cot_raw                | markowitz |  30 |           0.8348 |          0.0278 |         0.0656 |          23.9636 |         20.7240 |         49.3659 |           1374.7333 |            1580.1000 |
| claude_sonnet_raw   | zero_shot_raw          | markowitz |  30 |           0.7028 |          0.0234 |         0.0270 |          21.5433 |         21.3545 |         24.0688 |           1277.7333 |            1306.2333 |
| deterministic_opt   | reference              | markowitz |  30 |           0.0000 |          0.0000 |         0.0000 |           0.0129 |          0.0116 |          0.0242 |              0.0000 |               0.0000 |
| gemini_raw          | cot_raw                | markowitz |  30 |           1.0367 |          0.0346 |         0.0456 |          24.3787 |         22.7629 |         30.5142 |           2659.6333 |            3123.2333 |
| gemini_raw          | zero_shot_raw          | markowitz |  30 |           1.1132 |          0.0371 |         0.0402 |          22.9337 |         22.8906 |         26.5771 |           2571.6333 |            3389.0667 |
| claude_sonnet_agent | zero_shot_agent_cold   | tsp_tw    |  20 |          10.1118 |          0.5056 |         0.7110 |         278.5845 |        246.4692 |        408.5442 |          20167.4000 |           29672.5000 |
| claude_sonnet_agent | zero_shot_agent_guided | tsp_tw    |  20 |           7.2525 |          0.3626 |         0.5560 |         192.7305 |        215.5557 |        312.5179 |          11612.6000 |           21852.5000 |
| claude_sonnet_raw   | cot_raw                | tsp_tw    |  20 |           0.4852 |          0.0243 |         0.0327 |          21.6288 |         19.0380 |         31.4232 |            669.5500 |            1483.5000 |
| claude_sonnet_raw   | zero_shot_raw          | tsp_tw    |  20 |           0.4233 |          0.0212 |         0.0242 |          19.8414 |         19.5065 |         24.2868 |            558.5500 |            1299.4000 |
| deterministic_opt   | reference              | tsp_tw    |  20 |           0.0000 |          0.0000 |         0.0000 |           0.3019 |          0.2286 |          0.7111 |              0.0000 |               0.0000 |
| gemini_raw          | cot_raw                | tsp_tw    |  20 |           1.1237 |          0.0562 |         0.0830 |          38.0284 |         32.9038 |         55.1855 |            918.9500 |            5503.6500 |
| gemini_raw          | zero_shot_raw          | tsp_tw    |  20 |           0.7182 |          0.0359 |         0.0829 |          25.1252 |         20.5044 |         55.0380 |            817.9500 |            3488.8500 |

