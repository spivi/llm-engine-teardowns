"""Claude Sonnet provider — raw LLM, no tool calling.

Parses a JSON answer (weights or tour) from the model response.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from anthropic import Anthropic, APIStatusError, RateLimitError

from providers.base import Provider, ProviderResponse


MAX_TOKENS_PER_CALL = 4096


class ClaudeSonnetRawProvider(Provider):
    name = "claude_sonnet_raw"

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
        delay = 1.0
        last_err = None
        for _ in range(6):
            t0 = time.perf_counter()
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=MAX_TOKENS_PER_CALL,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                text = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", "") == "text"
                )
                in_tok = int(resp.usage.input_tokens)
                out_tok = int(resp.usage.output_tokens)
                cost = (
                    in_tok * self.price["input_per_mtok"] / 1_000_000
                    + out_tok * self.price["output_per_mtok"] / 1_000_000
                )
                return ProviderResponse(
                    raw_text=text,
                    latency_ms=elapsed_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
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


def _resolve_price(prices: Dict, model: str) -> Dict[str, float]:
    candidates = [k for k in prices if not k.startswith("_") and model.startswith(k)]
    if not candidates:
        raise ValueError(f"No price entry for model {model}; edit prices.json")
    key = max(candidates, key=len)
    return prices[key]
