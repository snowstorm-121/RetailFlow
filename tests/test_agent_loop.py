import asyncio
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from conftest import STATE_REF


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, **kwargs):
        self.inputs.append(messages)
        return self.responses.pop(0)


def call(name, args, ident):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": ident}])


def runtime(tmp_path, responses, **limits):
    from retailflow.runtime import RetailRuntime

    model = ScriptedModel(responses)
    worker = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: model)
    config = {"initial_state_ref": STATE_REF, "max_model_calls_per_turn": 6, **limits}
    return worker, model, config


def test_single_tool_result_and_answer_keep_call_id(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        AIMessage(content="Found your account."),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find my account: Yara Muller, ZIP 85041."}, config))
    assert result["status"] == "answered"
    assert len(model.bound_tools) == 9
    messages = worker.load_messages("a")
    tool = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool.tool_call_id == "u1" and tool.name == "find_user_id_by_name_zip"
    assert json.loads(tool.content)["data"] == "yara_muller_8652"
    assert result["usage"]["model_calls"] == 2
    assert result["usage"]["state_check_seconds"] >= 0


def test_two_dependent_tools_use_previous_result(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        AIMessage(content="Your order is pending."),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller, 85041. Check #W5056519"}, config))
    assert result["status"] == "answered"
    messages = worker.load_messages("a")
    assert [(m.name, m.tool_call_id) for m in messages if isinstance(m, ToolMessage)] == [
        ("find_user_id_by_name_zip", "u1"), ("get_order_details", "o1")
    ]
    assert any(isinstance(m, ToolMessage) and m.tool_call_id == "u1" for m in model.inputs[1])


def test_ask_user_pauses_and_next_message_resumes(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity"], "question": "What is your ZIP code?"}, "ask1"),
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        AIMessage(content="Found your account."),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "I am Yara Muller."}, config))
    assert first["status"] == "needs_input" and first["request"]["fields"] == ["identity"]
    second = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "85041"}, config))
    assert second["status"] == "answered"
    assert len([m for m in worker.load_messages("a") if isinstance(m, ToolMessage)]) == 2


def test_ask_user_schema_rejects_duplicate_requested_fields(tmp_path):
    from react_agent.tools import RETAIL_TOOLS

    schema = next(tool for tool in RETAIL_TOOLS if tool["function"]["name"] == "ask_user")
    assert schema["function"]["parameters"]["properties"]["fields"]["uniqueItems"] is True
    worker, _, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity", "identity"], "question": "ZIP?"}, "a1"),
        AIMessage(content="The duplicate field request was rejected."),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find me"}, config))
    assert result["status"] == "answered"
    assert json.loads(worker.load_messages("a")[-2].content)["code"] == "INVALID_ARGUMENT"


def test_multiple_calls_are_rejected_without_partial_execution(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content="", tool_calls=[
            {"name": "find_user_id_by_name_zip", "args": {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "id": "u1"},
            {"name": "cancel_pending_order", "args": {"order_id": "#W5056519", "reason": "ordered by mistake"}, "id": "c1"},
        ]),
        AIMessage(content="I need to handle one step at a time."),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Cancel #W5056519"}, config))
    assert result["status"] == "answered"
    tools = [m for m in worker.load_messages("a") if isinstance(m, ToolMessage)]
    assert len(tools) == 2 and all(m.status == "error" for m in tools)
    assert worker.environment.pending_proposal("a") is None
    assert worker.environment.get_order("a", "#W5056519")["status"] == "pending"


def test_tool_narration_does_not_bypass_identity_or_write_confirmation(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content="I cancelled it.", tool_calls=[{"name": "cancel_pending_order",
            "args": {"order_id": "#W5056519", "reason": "ordered by mistake"}, "id": "c1"}]),
        call("ask_user", {"fields": ["identity"], "question": "Your identity?"}, "ask2"),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Cancel #W5056519"}, config))
    assert result["status"] == "needs_input"
    tool = next(m for m in worker.load_messages("a") if isinstance(m, ToolMessage))
    assert tool.tool_call_id == "c1" and tool.status == "error"
    assert json.loads(tool.content)["code"] == "AUTH_REQUIRED"
    assert result["usage"]["protocol_errors"] == 0
    assert worker.environment.get_order("a", "#W5056519")["status"] == "pending"
    assert worker.environment.pending_proposal("a") is None


def test_confirmation_executes_exact_pending_action(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
        AIMessage(content="The order has been cancelled."),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "I am Yara Muller, 85041. Cancel #W5056519; I ordered by mistake."}, config))
    assert first["status"] == "needs_confirmation"
    assert worker.environment.get_order("a", "#W5056519")["status"] == "pending"
    second = asyncio.run(worker.run_turn("a", {"kind": "confirmation", "proposal_id": first["request"]["proposal_id"], "accepted": True}, config))
    assert second["status"] == "answered"
    assert second["usage"]["tool_calls"] == 1 and second["usage"]["tool_seconds"] > 0
    trace = json.loads(Path(second["trace_path"]).read_text())
    assert trace["elapsed_seconds"] + 0.001 >= second["usage"]["model_seconds"] + second["usage"]["tool_seconds"]
    assert worker.environment.get_order("a", "#W5056519")["status"] == "cancelled"
    assert len(worker.environment.get_order("a", "#W5056519")["payment_history"]) == 2


def test_model_call_budget_ends_cleanly(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
    ], max_model_calls_per_turn=1)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find Yara Muller 85041"}, config))
    assert result["status"] == "budget_exceeded"
    assert result["failure_reason"] == "model_call_budget"
    assert len([m for m in worker.load_messages("a") if isinstance(m, ToolMessage)]) == 1


def test_cli_chat_prints_status_and_confirmation_choice(tmp_path, monkeypatch, capsys):
    from retailflow import cli

    config_path = tmp_path / "local.json"
    config_path.write_text(json.dumps({"store_path": str(tmp_path / "business.sqlite"),
                                       "trace_dir": str(tmp_path / "traces")}))
    events = []

    async def scripted(thread, event, config):
        events.append(event)
        if len(events) == 1:
            return {"status": "needs_confirmation", "text": "Confirm cancellation?",
                    "request": {"proposal_id": "p1"}, "trace_path": "trace.json", "usage": {}, "failure_reason": None}
        return {"status": "answered", "text": "Cancelled.", "request": None,
                "trace_path": "trace2.json", "usage": {}, "failure_reason": None}

    monkeypatch.setattr(cli, "run_turn", scripted, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert asyncio.run(cli.chat("demo", config_path, "Cancel my order")) == 0
    assert events == [{"kind": "message", "text": "Cancel my order"},
                      {"kind": "confirmation", "proposal_id": "p1", "accepted": True}]
    output = capsys.readouterr().out
    assert "needs_confirmation" in output and "Cancelled." in output


def test_final_cannot_claim_uncreated_proposal(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        AIMessage(content="A cancellation proposal has been created for your order."),
        AIMessage(content="A cancellation proposal has been created for your order."),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Cancel #W5056519. I am Yara Muller, 85041."}, config))
    assert result["status"] == "failed"
    assert result["failure_reason"] == "proposal_not_created"
    assert worker.environment.pending_proposal("a") is None


def test_handoff_marker_does_not_claim_a_human_was_contacted(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("transfer_to_human_agents", {"summary": "Customer requests an exchange beyond this app."}, "h1"),
    ])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "I need an exchange, I am Yara Muller, 85041."}, config))
    assert result["status"] == "handed_off"
    assert "no human" in result["text"].lower()
