"""Utility & helper functions."""

import json
from pathlib import Path

import httpx
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama

from react_agent.context import Context


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(context: Context) -> ChatOllama:
    """Use the project's local Ollama endpoint."""
    async_kwargs = {"timeout": context.request_timeout_seconds}
    if context.trace_path:
        async_kwargs["event_hooks"] = make_http_hooks(Path(context.trace_path))
    return ChatOllama(
        model=context.model,
        base_url=context.base_url,
        temperature=context.temperature,
        seed=context.seed,
        num_ctx=context.num_ctx,
        num_predict=context.num_predict,
        client_kwargs={"timeout": context.request_timeout_seconds},
        async_client_kwargs=async_kwargs,
    )


def make_http_hooks(path: Path) -> dict:
    """Record the actual Ollama HTTP JSON on this single-threaded smoke run."""
    def append(kind: str, body: bytes, url: str, status: int | None = None) -> None:
        try:
            value = json.loads(body)
        except ValueError:
            value = body.decode(errors="replace")
        with path.open("a") as log:
            log.write(json.dumps({"kind": kind, "url": url, "status": status, "body": value}) + "\n")

    async def request_hook(request: httpx.Request) -> None:
        append("request", request.content, str(request.url))

    async def response_hook(response: httpx.Response) -> None:
        append("response", await response.aread(), str(response.request.url), response.status_code)

    return {"request": [request_hook], "response": [response_hook]}
