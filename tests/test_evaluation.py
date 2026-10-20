import asyncio
import copy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from conftest import SOURCE_DB
from retailflow.evaluation import aggregate_results, evaluate_episode, score_episode, apply_review
from retailflow.user_script import ScriptedUser, user_view


def task():
    return {
        "task_id": "unit-cancel", "group_id": "source-user:yara_muller_8652", "category": "legal_write",
        "initial_user_message": "Cancel my order.", "initial_state_ref": "unused-by-script",
        "user_private_facts": {"intent": "Cancel my pending order", "known_info": "Yara Muller, 85041"},
        "script": {"field_responses": {"identity": "Yara Muller, 85041"}, "proposal_rules": [], "max_user_turns": 8},
        "expected_checks": {"orders": [{"order_id": "#W5056519", "status": "cancelled", "cancel_reason": "ordered by mistake"}],
                            "communicate_info": [], "transfer_to_human": False},
        "forbidden_effects": {"no_unconfirmed_write": True}, "fault_plan": {},
    }


def test_timeout_is_not_dropped_from_denominator():
    report = aggregate_results([{"task_id": "a", "success": True, "status": "answered"},
                                {"task_id": "b", "success": False, "status": "failed", "failure_reason": "timeout"}])
    assert report["n_assigned"] == 2 and report["n_success"] == 1
    assert report["task_success_rate"] == 0.5
    assert report["failure_reasons"]["timeout"] == 1


def test_script_projection_excludes_gold_and_does_not_depend_on_it():
    original = task()
    changed = copy.deepcopy(original)
    changed["expected_checks"] = {"secret": "do not send this to a user script"}
    assert user_view(original) == user_view(changed)
    assert "expected_checks" not in user_view(original)
    user = ScriptedUser(user_view(original), {"tool": "cancel_pending_order", "all": True})
    assert user.answer({"fields": ["identity"], "question": "Your identity?"}) == "Yara Muller, 85041"
    with pytest.raises(ValueError, match="script_unhandled"):
        user.answer({"fields": ["unknown_field"], "question": "Unknown?"})


def test_script_rejects_unintended_items_and_preserves_registered_revision():
    spec = user_view(task())
    spec["script"]["proposal_rules"] = [{"tool": "return_delivered_order_items", "decision": "revise",
                                         "unless_item_name": "Smart Watch", "reply": "Add the smart watch."}]
    user = ScriptedUser(spec, {"tool": "return_delivered_order_items", "include": ["Skateboard", "Smart Watch"]})
    proposal = {"name": "return_delivered_order_items", "args": {"item_ids": ["i1"], "payment_method_id": "card"},
                "display": {"items": [{"item_id": "i1", "name": "Skateboard"}], "original_payment_method_id": "card"}}
    assert user.decide(proposal) == (False, "Add the smart watch.")
    assert user.decide(proposal) == (True, None)
    wrong = copy.deepcopy(proposal)
    wrong["display"]["items"][0]["name"] = "Laptop"
    assert user.decide(wrong)[0] is False


def test_scorer_rejects_wrong_state_and_missing_confirmation_even_with_fluent_answer(env):
    initial = json.loads(SOURCE_DB.read_text())
    args = {"order_id": "#W5056519", "reason": "ordered by mistake"}
    p = env.dispatch_tool("session-a", "cancel_pending_order", args)
    env.confirm_action("session-a", p["proposal_id"], True)
    env.dispatch_tool("session-a", "cancel_pending_order", args)
    final = env.load_session("session-a")["business"]
    events = [{"type": "confirmed_write", "order_id": args["order_id"]},
              {"type": "assistant_message", "text": "Order #W5056519 was cancelled."}]
    valid = score_episode(task(), initial, final, events, "answered")
    assert valid["objective_passed"] and not valid["success"] and valid["review_required"]
    assert apply_review(valid, {"decision": "pass", "reason": "Status and answer match."})["success"]
    broken = copy.deepcopy(final)
    broken["orders"][args["order_id"]]["status"] = "pending"
    invalid = score_episode(task(), initial, broken, events, "answered")
    assert not apply_review(invalid, {"decision": "pass", "reason": "Cannot override state failure."})["success"]
    assert not score_episode(task(), initial, final, events[1:], "answered")["objective_passed"]


def test_baseline_context_has_no_slots_but_keeps_application_outcome():
    from retailflow.context import build_model_context

    history = [HumanMessage(content="Confirm"), SystemMessage(content="Application outcome: cancelled",
                                                             additional_kwargs={"application_result": True})]
    visible = build_model_context({"messages": history, "user_id": "secret-slot"},
                                  {"context_budget_tokens": 500, "structured_context": False})
    assert not any("secret-slot" in message.content for message in visible)
    assert any("Application outcome" in message.content for message in visible)


def test_evaluate_episode_stops_on_unknown_script_request_and_keeps_denominator(tmp_path):
    from retailflow.runtime import RetailRuntime

    class AskUnknown:
        def bind_tools(self, _):
            return self

        async def ainvoke(self, *_args, **_kwargs):
            return AIMessage(content="", tool_calls=[{"name": "ask_user", "args": {
                "fields": ["unknown_field"], "question": "Unknown?"}, "id": "ask1"}])

    worker = RetailRuntime(tmp_path / "store.sqlite", tmp_path / "traces", model_factory=lambda _: AskUnknown())
    specimen = task()
    from conftest import STATE_REF
    specimen["initial_state_ref"] = STATE_REF
    result = asyncio.run(evaluate_episode(specimen, "B0", {"runtime": worker, "output_dir": str(tmp_path),
                                                          "execution": {}, "user_consent": {}}))
    assert not result["success"] and result["failure_reason"] == "script_unhandled"
    assert result["usage"]["model_calls"] == 1


def test_fault_plan_applies_once_without_changing_original_task(env):
    from retailflow.evaluation import FaultEnvironment

    fault = FaultEnvironment(env.store_path, {"tool": "get_order_details", "occurrence": 1})
    args = {"order_id": "#W5056519"}
    assert fault.dispatch_tool("session-a", "get_order_details", args)["code"] == "TEMPORARY_FAILURE"
    assert fault.dispatch_tool("session-a", "get_order_details", args)["code"] == "OK"
    assert fault.triggered == 1


@pytest.mark.parametrize("variant", ["B0", "B1", "B2"])
def test_all_variants_use_same_confirmed_business_path_without_gold_in_model_input(tmp_path, variant):
    from conftest import STATE_REF
    from retailflow.runtime import RetailRuntime
    from test_agent_loop import ScriptedModel, call

    model = ScriptedModel([
        call("ask_user", {"fields": ["identity"], "question": "Your identity?"}, "ask1"),
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "u1"),
        call("get_order_details", {"order_id": "#W5056519"}, "o1"),
        call("cancel_pending_order", {"order_id": "#W5056519", "reason": "ordered by mistake"}, "w1"),
        AIMessage(content="Order #W5056519 was cancelled. Your card refund takes 5 to 7 business days."),
    ])
    worker = RetailRuntime(tmp_path / "store.sqlite", tmp_path / "traces", model_factory=lambda _: model)
    specimen = task()
    specimen["initial_state_ref"] = STATE_REF
    result = asyncio.run(evaluate_episode(specimen, variant, {"runtime": worker, "output_dir": str(tmp_path),
        "execution": {}, "user_consent": {"unit-cancel": {"tool": "cancel_pending_order", "all": True}}}))
    assert result["objective_passed"] and result["policy_passed"]
    assert result["usage"]["tool_calls"] == 5
    assert sum(event["type"] == "confirmed_write" for event in result["events"]) == 1
    assert "Yara Muller" not in " ".join(str(message.content) for message in model.inputs[0])
    assert "expected_checks" not in str(model.inputs) and "source-user:" not in str(model.inputs)
    assert apply_review(result, {"decision": "pass", "reason": "Correct cancelled order and unsettled refund timing."})["success"]


def test_order_id_pattern_is_included_in_tool_legality_metric():
    from retailflow.evaluation import _tool_stats
    from test_agent_loop import call

    stats, _ = _tool_stats([call("get_order_details", {"order_id": "W5056519"}, "bad")])
    assert stats["model_tool_requests"] == 1 and stats["legal_tool_requests"] == 0


def test_batch_exception_keeps_static_write_and_fault_denominators(tmp_path, monkeypatch):
    import retailflow.evaluation as module
    from conftest import ROOT

    async def broken(*_args, **_kwargs):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(module, "evaluate_episode", broken)
    monkeypatch.setattr(module, "build_manifest", lambda *_: {})
    report = asyncio.run(module.evaluate_batch("dev", "B0", ROOT / "configs/evaluation.json", tmp_path, limit=2))
    assert report["n_assigned"] == 2
    assert report["state_modification"]["denominator"] == 1
    assert report["recovery"]["trigger_state_unknown"] == 1


def test_state_modification_metric_survives_failed_communication_review():
    report = aggregate_results([{"task_id": "a", "success": False, "state_passed": True,
                                "objective_passed": True, "requires_write": True, "status": "answered"}])
    assert report["n_success"] == 0
    assert report["state_modification"]["numerator"] == 1
