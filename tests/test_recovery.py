import asyncio

import pytest

from langchain_core.messages import AIMessage

from test_agent_loop import call, runtime


def test_temporary_read_failure_retries_once_when_enabled(tmp_path, monkeypatch):
    import retailflow.environment as module

    original = module.invoke
    attempts = []

    def transient(db, name, args):
        if name == "get_order_details":
            attempts.append(args)
            if len(attempts) == 1:
                raise module.PreExecutionTemporaryFailure("read unavailable before execution")
        return original(db, name, args)

    monkeypatch.setattr(module, "invoke", transient)
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        AIMessage(content="Order found."),
    ], max_retry_per_call=1)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041. Check #W5056519"}, config))
    assert result["status"] == "answered"
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert result["usage"]["retries"] == 1
    attempts_in_trace = [attempt for attempt in result["usage"]["tool_attempts"] if attempt["name"] == "get_order_details"]
    assert [attempt["code"] for attempt in attempts_in_trace] == ["TEMPORARY_FAILURE", "OK"]


def test_temporary_read_failure_stays_visible_without_retry(tmp_path, monkeypatch):
    import retailflow.environment as module

    original = module.invoke
    attempts = []

    def transient(db, name, args):
        if name == "get_order_details":
            attempts.append(args)
            raise module.PreExecutionTemporaryFailure("read unavailable before execution")
        return original(db, name, args)

    monkeypatch.setattr(module, "invoke", transient)
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        AIMessage(content="The read failed."),
    ], max_retry_per_call=0)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041. Check #W5056519"}, config))
    assert result["status"] == "answered"
    assert len(attempts) == 1
    assert result["usage"]["retries"] == 0
    assert "TEMPORARY_FAILURE" in worker.load_messages("a")[-2].content


def test_policy_denial_is_not_retried(tmp_path, monkeypatch):
    from retailflow.environment import RetailEnvironment

    original = RetailEnvironment.dispatch_tool
    seen = []

    def counted(self, thread, name, args):
        if name == "get_order_details":
            seen.append(args)
        return original(self, thread, name, args)

    monkeypatch.setattr(RetailEnvironment, "dispatch_tool", counted)
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W1679211"}, "o1"),
    ], max_retry_per_call=1)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041. Check #W1679211"}, config))
    assert result["status"] == "refused"
    assert len(seen) == 1
    assert result["usage"]["retries"] == 0


def test_model_timeout_preserves_completed_tool_messages_and_counts_attempt(tmp_path):
    from retailflow.runtime import RetailRuntime

    class TimeoutModel:
        calls = 0

        def bind_tools(self, _):
            return self

        async def ainvoke(self, messages, **_):
            self.calls += 1
            if self.calls == 1:
                return call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1")
            raise TimeoutError("injected model timeout")

    model = TimeoutModel()
    worker = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: model)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find Yara Muller 85041"}, {}))
    assert result["status"] == "failed" and result["failure_reason"] == "timeout"
    assert worker.load_messages("a")[-1].tool_call_id == "u1"
    assert result["usage"]["model_calls"] == 2
    assert worker.load_session("a")["runtime"]["model_calls"] == 2


def test_episode_model_budget_survives_runtime_reopen(tmp_path):
    from retailflow.runtime import RetailRuntime

    worker, model, config = runtime(tmp_path, [AIMessage(content="Hello.")], max_model_calls_per_episode=1)
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Hello"}, config))
    assert first["status"] == "answered"
    reopened = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: model)
    second = asyncio.run(reopened.run_turn("a", {"kind": "message", "text": "Hello again"}, config))
    assert second["status"] == "budget_exceeded"
    assert second["failure_reason"] == "episode_model_call_budget"
    assert len(model.inputs) == 1


def test_truncated_model_response_does_not_execute_tool(tmp_path):
    response = call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1")
    response.response_metadata = {"done_reason": "length"}
    worker, _, config = runtime(tmp_path, [response])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find me"}, config))
    assert result["status"] == "failed" and result["failure_reason"] == "output_truncated"
    assert result["usage"]["tool_calls"] == 0


def test_new_request_revokes_confirmation_saved_before_unexecuted_write(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c2"),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519"}, config))
    old_id = first["request"]["proposal_id"]
    assert worker.environment.confirm_action("a", old_id, True)["ok"]  # crash before dispatch
    revised = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Actually keep that order."}, config))
    assert revised["status"] == "needs_confirmation"
    assert revised["request"]["proposal_id"] != old_id
    assert worker.environment.get_order("a", "#W5056519")["status"] == "pending"


@pytest.mark.parametrize("limits,reason", [
    ({"max_user_turns_per_episode": 0}, "episode_user_turn_budget"),
    ({"max_episode_seconds": 0}, "episode_time_budget"),
    ({"context_budget_tokens": 1}, "context_budget"),
])
def test_exhausted_budget_stops_before_model_request(tmp_path, limits, reason):
    worker, model, config = runtime(tmp_path, [], **limits)
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Hello"}, config))
    assert result["status"] == "budget_exceeded" and result["failure_reason"] == reason
    assert model.inputs == []


def test_model_timeout_after_write_reports_committed_business_result(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519"}, config))

    class TimeoutModel:
        def bind_tools(self, _):
            return self

        async def ainvoke(self, *_, **__):
            raise TimeoutError("reply timed out")

    worker.model_factory = lambda _: TimeoutModel()
    result = asyncio.run(worker.run_turn("a", {"kind": "confirmation", "proposal_id": first["request"]["proposal_id"],
                                                 "accepted": True}, config))
    assert result["failure_reason"] == "timeout"
    assert "committed" in result["text"].lower()
    assert worker.environment.get_order("a", "#W5056519")["status"] == "cancelled"


def test_actual_input_over_reserved_limit_stops_before_tool_execution(tmp_path):
    response = call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1")
    response.usage_metadata = {"input_tokens": 8192, "output_tokens": 10, "total_tokens": 8202}
    worker, _, config = runtime(tmp_path, [response])
    result = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Find me"}, config))
    assert result["status"] == "budget_exceeded"
    assert result["failure_reason"] == "actual_context_limit"
    assert result["usage"]["tool_calls"] == 0


def test_replayed_confirmation_after_committed_write_does_not_write_twice(tmp_path):
    from retailflow.runtime import RetailRuntime

    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "c1"),
    ])
    first = asyncio.run(worker.run_turn("a", {"kind": "message", "text": "Yara Muller 85041; cancel #W5056519, ordered by mistake"}, config))
    proposal = worker.environment.pending_proposal("a")
    worker.environment.confirm_action("a", proposal["proposal_id"], True)
    result = worker.environment.dispatch_tool("a", proposal["name"], proposal["args"])
    assert result["ok"]
    before = worker.environment.get_order("a", "#W5056519")

    class FinalModel:
        def bind_tools(self, _):
            return self

        async def ainvoke(self, messages, **_):
            return AIMessage(content="The confirmed order cancellation is recorded.")

    reopened = RetailRuntime(tmp_path / "business.sqlite", tmp_path / "traces", model_factory=lambda _: FinalModel())
    replay = asyncio.run(reopened.run_turn("a", {"kind": "confirmation", "proposal_id": first["request"]["proposal_id"],
                                                    "accepted": True}, config))
    assert replay["status"] == "answered"
    assert replay["usage"]["tool_calls"] == 0
    assert reopened.environment.get_order("a", "#W5056519") == before
    assert len(before["payment_history"]) == 2
