"""Provider interface for the experiment.

Non-agent providers can ignore the `instance` argument; the agent provider needs
it for self-validation between attempts. The response carries everything the
runner needs to write one CSV row.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentToolCall:
    """One tool invocation in the agent loop."""
    attempt: int
    turn: int
    tool: str
    args_summary: str           # truncated repr of args (for readability)
    status: str                 # the tool's "status" field
    objective: Optional[float]
    message: str                # the tool's "message" field, truncated


@dataclass
class ProviderResponse:
    """Result of one provider call on one instance."""
    raw_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    attempt_number: int = 1
    tool_calls: List[AgentToolCall] = field(default_factory=list)
    final_answer_used_tools: bool = False
    early_error_type: Optional[str] = None


class Provider(ABC):
    """Provider interface.

    Implementations may ignore `instance` if they don't need it (e.g., raw LLM
    providers just format the prompt and return the LLM response).
    """
    name: str = ""

    @abstractmethod
    def call(
        self,
        prompt: str,
        instance: Dict[str, Any],
        temperature: float,
    ) -> ProviderResponse:
        ...
