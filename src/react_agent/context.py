"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen3:4b-instruct-2507-q4_K_M",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Local Ollama model tag."
        },
    )

    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 768
    request_timeout_seconds: int = 120
    trace_path: str | None = None

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))
