"""Gemini provider — raw LLM, no tool calling.

Counterpart to claude_sonnet_raw. Uses the `google-genai` SDK.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from providers.base import Provider, ProviderResponse


MAX_TOKENS_PER_CALL = 8192
THINKING_BUDGET = 2048   # Gemini 2.5 Pro requires thinking; cap to bound cost.


class GeminiRawProvider(Provider):
    name = "gemini_raw"

    def __init__(self, prices_path: str = "prices.json"):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. "
                "Add it to .env before running gemini_raw."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
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
        last_err: Exception | None = None
        cfg = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=MAX_TOKENS_PER_CALL,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
        )
        for _ in range(6):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=cfg,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                text = resp.text or ""
                usage = getattr(resp, "usage_metadata", None)
                in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
                visible = int(getattr(usage, "candidates_token_count", 0) or 0)
                # Gemini bills "thinking" tokens at the output rate; include them.
                thoughts = int(getattr(usage, "thoughts_token_count", 0) or 0)
                out_tok = visible + thoughts
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
            except genai_errors.APIError as e:
                last_err = e
                code = getattr(e, "code", 0) or getattr(e, "status_code", 0)
                if code in (429, 500, 502, 503, 504):
                    time.sleep(delay)
                    delay = min(delay * 2, 32)
                else:
                    raise
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 32)
        raise RuntimeError(f"Gemini API: retries exhausted: {last_err}")


def _resolve_price(prices: Dict, model: str) -> Dict[str, float]:
    candidates = [k for k in prices if not k.startswith("_") and model.startswith(k)]
    if not candidates:
        raise ValueError(f"No price entry for model {model}; edit prices.json")
    key = max(candidates, key=len)
    return prices[key]
