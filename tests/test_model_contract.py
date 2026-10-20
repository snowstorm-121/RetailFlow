import asyncio
import importlib
import json

import httpx
import pytest
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, ToolMessage


def run_scripted_call(monkeypatch, call):
    module = importlib.import_module("react_agent.graph")

    class ScriptedModel:
        def __init__(self):
            self.calls = 0

        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[call])
            return AIMessage(content="done")

    model = ScriptedModel()
    monkeypatch.setattr(module, "load_chat_model", lambda context: model)
    result = asyncio.run(
        module.graph.ainvoke(
            {"messages": [{"role": "user", "content": "test"}]},
            context=module.Context(),
            config={"recursion_limit": 8},
        )
    )
    return result["messages"]


def test_template_uses_only_local_smoke_tools():
    from react_agent.tools import TOOLS

    assert {tool.name for tool in TOOLS} == {
        "add",
        "lookup_route_minutes",
        "minutes_to_seconds",
    }


def test_model_loader_binds_local_ollama():
    from react_agent.context import Context
    from react_agent.utils import load_chat_model

    context = Context(
        model="qwen3:4b-instruct-2507-q4_K_M",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=8192,
        num_predict=768,
        request_timeout_seconds=120,
    )
    model = load_chat_model(context)
    assert isinstance(model, ChatOllama)
    assert model.model == context.model


def test_unknown_tool_gets_matching_error_message(monkeypatch):
    call = {"name": "delete_everything", "args": {}, "id": "call-unknown"}
    messages = run_scripted_call(monkeypatch, call)
    message = next(m for m in messages if isinstance(m, ToolMessage))
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == call["id"]
    assert message.status == "error"


def test_missing_argument_gets_matching_error_message(monkeypatch):
    call = {"name": "add", "args": {"a": 2}, "id": "call-missing"}
    messages = run_scripted_call(monkeypatch, call)
    message = next(m for m in messages if isinstance(m, ToolMessage))
    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == call["id"]
    assert message.status == "error"


def test_valid_tool_preserves_name_arguments_and_id(monkeypatch):
    call = {"name": "add", "args": {"a": 2, "b": 3}, "id": "call-add"}
    messages = run_scripted_call(monkeypatch, call)
    request = next(m for m in messages if isinstance(m, AIMessage) and m.tool_calls)
    assert request.tool_calls[0]["args"] == call["args"]
    message = next(m for m in messages if isinstance(m, ToolMessage))
    assert isinstance(message, ToolMessage)
    assert message.name == call["name"]
    assert message.tool_call_id == call["id"]
    assert message.content == "5"


def test_http_hooks_record_native_request_and_response(tmp_path):
    from react_agent.utils import make_http_hooks

    path = tmp_path / "http.jsonl"
    hooks = make_http_hooks(path)
    request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat", json={"model": "local", "tools": []})
    response = httpx.Response(200, request=request, json={"message": {"tool_calls": []}})
    asyncio.run(hooks["request"][0](request))
    asyncio.run(hooks["response"][0](response))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["body"]["model"] == "local"
    assert records[0]["url"] == "http://127.0.0.1:11434/api/chat"
    assert records[1]["body"]["message"]["tool_calls"] == []


def test_disconnected_local_model_fails_without_fallback():
    from react_agent.context import Context
    from react_agent.graph import graph

    with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
        asyncio.run(graph.ainvoke(
            {"messages": [{"role": "user", "content": "Say hello."}]},
            context=Context(base_url="http://127.0.0.1:1", request_timeout_seconds=2),
        ))
