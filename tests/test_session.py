import asyncio
import subprocess
import sys

from langchain_core.messages import AIMessage

from conftest import STATE_REF
from test_agent_loop import call, runtime


def test_load_session_restores_messages_identity_target_and_pending_after_reopen(tmp_path):
    from retailflow.runtime import RetailRuntime

    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519, ordered by mistake"}, config))
    assert first["status"] == "needs_confirmation"
    reopened = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: None)
    session = reopened.load_session("a")
    assert session["user_id"] == "yara_muller_8652"
    assert session["target_order_id"] == "#W5056519"
    assert session["target_order_status"] == "pending"
    assert session["pending_action"]["proposal_id"] == first["request"]["proposal_id"]
    assert len(session["messages"]) == len(worker.load_messages("a"))
    assert session["business"]["orders"]["#W5056519"]["status"] == "pending"
    assert reopened.load_session("other") == {}


def test_confirmed_action_can_continue_after_runtime_reopen(tmp_path):
    from retailflow.runtime import RetailRuntime

    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519, ordered by mistake"}, config))
    assert first["status"] == "needs_confirmation"

    class FinalModel:
        def bind_tools(self, _):
            return self

        async def ainvoke(self, messages, **_):
            return AIMessage(content="The order has been cancelled.")

    reopened = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: FinalModel())
    second = asyncio.run(reopened.run_turn("a", {"kind": "confirmation", "proposal_id": first["request"]["proposal_id"],
                                                    "accepted": True}, config))
    assert second["status"] == "answered"
    assert reopened.environment.get_order("a", "#W5056519")["status"] == "cancelled"


def test_cli_reopens_pending_proposal_before_accepting_new_message(tmp_path, monkeypatch, capsys):
    import json
    from retailflow import cli
    from retailflow.environment import RetailEnvironment

    store = tmp_path / "business.sqlite"
    env = RetailEnvironment(store)
    env.create_session("a", STATE_REF)
    env.dispatch_tool("a", "find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"})
    proposal = env.dispatch_tool("a", "cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"})
    config_path = tmp_path / "local.json"
    config_path.write_text(json.dumps({"store_path": str(store), "trace_dir": str(tmp_path / "traces")}))
    events = []

    async def captured(thread, event, _):
        events.append((thread, event))
        return {"status": "answered", "text": "Recorded.", "trace_path": "trace.json", "request": None}

    answers = iter(["yes", "/quit"])
    monkeypatch.setattr(cli, "run_turn", captured)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert asyncio.run(cli.chat("a", config_path)) == 0
    assert events == [("a", {"kind": "confirmation", "proposal_id": proposal["proposal_id"], "accepted": True})]
    assert "pending" in capsys.readouterr().out.lower()


def test_separate_python_process_resumes_pending_write(tmp_path):
    store = tmp_path / "business.sqlite"
    create = f'''from retailflow.environment import RetailEnvironment
from retailflow.upstream.paths import RETAIL_DB_PATH
import hashlib
env = RetailEnvironment({str(store)!r})
env.create_session("restart", "retail-db@" + hashlib.sha256(RETAIL_DB_PATH.read_bytes()).hexdigest())
env.dispatch_tool("restart", "find_user_id_by_name_zip", {{"first_name":"Yara","last_name":"Muller","zip":"85041"}})
proposal = env.dispatch_tool("restart", "cancel_pending_order", {{"order_id":"#W5056519","reason":"ordered by mistake"}})
print(proposal["proposal_id"])
'''
    first = subprocess.run([sys.executable, "-c", create], capture_output=True, text=True, check=True)
    proposal_id = first.stdout.strip()
    resume = f'''import asyncio
from langchain_core.messages import AIMessage
from retailflow.runtime import RetailRuntime
class Model:
    def bind_tools(self, tools): return self
    async def ainvoke(self, messages, **kwargs): return AIMessage(content="Cancelled.")
worker = RetailRuntime({str(store)!r}, {str(tmp_path / "traces")!r}, model_factory=lambda _: Model())
result = asyncio.run(worker.run_turn("restart", {{"kind":"confirmation","proposal_id":{proposal_id!r},"accepted":True}}, {{}}))
print(result["status"])
'''
    second = subprocess.run([sys.executable, "-c", resume], capture_output=True, text=True, check=True)
    assert second.stdout.strip() == "answered"
    from retailflow.environment import RetailEnvironment

    assert RetailEnvironment(store).get_order("restart", "#W5056519")["status"] == "cancelled"


def test_new_message_invalidates_unconfirmed_proposal(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
        AIMessage(content="Understood; please describe the new target."),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519, ordered by mistake"}, config))
    assert first["status"] == "needs_confirmation"
    asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Actually, I mean a different order."}, config))
    assert worker.environment.pending_proposal("a") is None
    old = asyncio.run(worker.run_turn("a", {"kind": "confirmation", "proposal_id": first["request"]["proposal_id"],
                                                 "accepted": True}, config))
    assert old["status"] == "failed"
    assert worker.environment.get_order("a", "#W5056519")["status"] == "pending"
