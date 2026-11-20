import asyncio

import pytest

from langchain_core.messages import AIMessage

from test_agent_loop import call, runtime


def test_address_text_confirmation_recovers_after_item_modification_status(tmp_path):
    from retailflow.upstream.paths import RETAIL_DB_PATH
    import json

    data = json.loads(RETAIL_DB_PATH.read_text())
    order = data["orders"]["#W8665881"]
    args = {"order_id": order["order_id"], **order["address"], "address2": "Suite 641"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Fatima", "last_name": "Johnson", "zip": "78712"}, "identity"),
        call("get_order_details", {"order_id": order["order_id"]}, "read"),
        AIMessage(content="Please confirm if you would like me to update the shipping address."),
        call("modify_pending_order_address", args, "propose"),
    ], isolated_proposal_repair=True)
    agent.environment.create_session("address", config["initial_state_ref"])
    with agent.environment.storage.transaction() as db:
        order["status"] = "pending (item modified)"
        db.execute("UPDATE sessions SET business_json=? WHERE thread=?", (json.dumps(data), "address"))
    result = asyncio.run(agent.run_turn("address", {
        "kind": "message", "text": "I'm Fatima Johnson, ZIP 78712. Change the shipping suite to Suite 641.",
    }, config))
    assert result["status"] == "needs_confirmation"
    assert result["request"]["args"] == args
    assert agent.environment.get_order("address", order["order_id"])["address"]["address2"] == "Suite 640"


def test_partial_address_answer_rechecks_remaining_goal_and_requires_new_confirmation(tmp_path):
    args = {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960",
            "city": "San Diego", "state": "CA", "country": "USA", "zip": "92101"}
    cancel = {"order_id": "#W5995614", "reason": "ordered by mistake"}
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("ask_user", {"fields": ["cancel_reason"], "question": "Why cancel the other order?"}, "reason"),
        call("modify_pending_order_address", args, "address"),
        AIMessage(content="The shipping address has been updated."),
        call("cancel_pending_order", cancel, "cancel"),
        AIMessage(content="The address was updated and the other order was cancelled."),
        AIMessage(content="I updated the shipping address and cancelled the other order."),
    ])
    async def run():
        first = await agent.run_turn("mixed", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change shipping for #W5056519 to 380 Maple Drive, Suite 960, "
            "San Diego, CA 92101, USA. Also cancel #W5995614."}, config)
        assert first["status"] == "needs_input"
        first = await agent.run_turn("mixed", {"kind": "message", "text": "ordered by mistake"}, config)
        second = await agent.run_turn("mixed", {"kind": "confirmation", "accepted": True,
                                               "proposal_id": first["request"]["proposal_id"]}, config)
        assert second["status"] == "needs_confirmation"
        assert second["request"]["proposal_id"] != first["request"]["proposal_id"]
        assert second["request"]["args"] == cancel
        retry = second["usage"]["context_inputs"][-1]
        assert {"modify_pending_order_items", "transfer_to_human_agents"}.isdisjoint(retry["tool_names"])
        assert "The shipping address has been updated." not in str(retry["messages"])
        assert "I explicitly confirm the displayed proposal." not in str(retry["messages"])
        assert '"next_step"' in retry["messages"][1]["data"]["content"]
        import json
        assert json.loads(retry["messages"][1]["data"]["content"])["customer_followups"] == ["ordered by mistake"]
        assert agent.environment.get_order("mixed", cancel["order_id"])["status"] == "pending"
        final = await agent.run_turn("mixed", {"kind": "confirmation", "accepted": True,
                                              "proposal_id": second["request"]["proposal_id"]}, config)
        assert final["status"] == "answered"
        assert agent.environment.get_order("mixed", cancel["order_id"])["status"] == "cancelled"
    asyncio.run(run())
    assert not model.responses


def test_optional_completion_check_preserves_last_budgeted_answer(tmp_path):
    args = {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960",
            "city": "San Diego", "state": "CA", "country": "USA", "zip": "92101"}
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("modify_pending_order_address", args, "address"),
        AIMessage(content="The address was updated. The other operation remains outstanding."),
    ], max_model_calls_per_episode=3)

    async def run():
        first = await agent.run_turn("budget", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change the address for #W5056519, and also cancel #W5995614."}, config)
        final = await agent.run_turn("budget", {"kind": "confirmation", "accepted": True,
                                               "proposal_id": first["request"]["proposal_id"]}, config)
        assert final["status"] == "answered"
        assert final["text"].endswith("remains outstanding.")
    asyncio.run(run())
    assert not model.responses


@pytest.mark.parametrize("claim", [
    "Cancellation of order #W5995614 was initiated.",
    "**#W5995614** canceled with refund of $164.28.",
])
def test_followup_cannot_report_cancellation_without_receipt(tmp_path, claim):
    args = {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960",
            "city": "San Diego", "state": "CA", "country": "USA", "zip": "92101"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("modify_pending_order_address", args, "address"),
        AIMessage(content="The address was updated."),
        AIMessage(content=claim),
    ])

    async def run():
        first = await agent.run_turn("mixed", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change the address in #W5056519 and cancel #W5995614."}, config)
        result = await agent.run_turn("mixed", {"kind": "confirmation", "accepted": True,
                                                "proposal_id": first["request"]["proposal_id"]}, config)
        assert result["status"] == "failed"
        assert result["failure_reason"] == "unverified_cancellation_claim"
        assert claim not in result["text"]
        assert agent.environment.get_order("mixed", "#W5995614")["status"] == "pending"
    asyncio.run(run())


def test_text_tool_name_and_arguments_envelope_is_not_a_final_answer(tmp_path):
    agent, _, config = runtime(tmp_path, [
        AIMessage(content='{"name": "cancel_pending_order", "arguments": {"order_id": "W5995614", "reason": "ordered by mistake"}}'),
    ])
    result = asyncio.run(agent.run_turn("text-tool", {"kind": "message", "text": "Cancel my order."}, config))
    assert result["status"] == "failed"
    assert result["failure_reason"] == "non_native_tool_action"
    assert result["usage"]["tool_calls"] == 0


def test_confirmed_address_focuses_next_explicit_step_and_rejects_wrong_target(tmp_path):
    address = {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960",
               "city": "San Diego", "state": "CA", "country": "USA", "zip": "92101"}
    cancel = {"order_id": "#W5995614", "reason": "ordered by mistake"}
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("modify_pending_order_address", address, "address"),
        call("modify_pending_order_address", address, "wrong-action"),
        call("cancel_pending_order", {**cancel, "order_id": address["order_id"]}, "wrong-target"),
        call("cancel_pending_order", cancel, "cancel"),
    ])

    async def run():
        proposal = await agent.run_turn("focus", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change shipping for #W5056519 to 380 Maple Drive, Suite 960, "
            "San Diego, CA 92101, USA. Also cancel #W5995614 because I ordered it by mistake."}, config)
        result = await agent.run_turn("focus", {"kind": "confirmation", "accepted": True,
                                                 "proposal_id": proposal["request"]["proposal_id"]}, config)
        assert result["status"] == "needs_confirmation"
        assert result["request"]["args"] == cancel
        assert agent.environment.get_order("focus", cancel["order_id"])["status"] == "pending"
        assert agent.environment.get_order("focus", address["order_id"])["status"] == "pending"
        assert any(attempt["code"] == "INVALID_ARGUMENT" for attempt in result["usage"]["tool_attempts"])
        first, retry, fallback = result["usage"]["context_inputs"]
        other_writes = {"modify_pending_order_address", "modify_pending_order_items", "modify_user_address"}
        assert set(first["tool_names"]).isdisjoint(other_writes)
        assert set(retry["tool_names"]).isdisjoint(other_writes)
        assert other_writes <= set(fallback["tool_names"])
        retry_message = retry["messages"][-1]["data"]["content"]
        assert "cancel_pending_order for #W5995614" in retry_message
    asyncio.run(run())


def test_ineligible_next_step_does_not_hide_other_writes(tmp_path):
    address = {"order_id": "#W5056519", "address1": "380 Maple Drive", "address2": "Suite 960",
               "city": "San Diego", "state": "CA", "country": "USA", "zip": "92101"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("modify_pending_order_address", address, "address"),
        AIMessage(content="Address updated. The delivered order is ineligible for cancellation."),
        AIMessage(content="Address updated. The delivered order is ineligible for cancellation."),
    ], isolated_proposal_repair=True)

    async def run():
        result = await agent.run_turn("focus", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change shipping for #W5056519 to San Diego. "
            "Also cancel #W9384736 because I ordered it by mistake."}, config)
        result = await agent.run_turn("focus", {"kind": "confirmation", "accepted": True,
                                                 "proposal_id": result["request"]["proposal_id"]}, config)
        assert result["status"] == "answered"
        assert "return_delivered_order_items" in result["usage"]["context_inputs"][0]["tool_names"]
    asyncio.run(run())


@pytest.mark.parametrize("label", ["credit card", "account"])
def test_confirmed_item_change_corrects_real_charge_wording(tmp_path, label):
    change = {"order_id": "#W5056519", "item_ids": ["7902309762"],
              "new_item_ids": ["1573035764"], "payment_method_id": "credit_card_3095586"}
    draft = f"The makeup kit was updated. Price difference: $10.36 (charged to your {label}). Credit card ending in 3095586."
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("modify_pending_order_items", change, "items"),
        AIMessage(content=draft),
    ])

    async def run():
        proposal = await agent.run_turn("payment", {"kind": "message", "text":
            "I'm Yara Muller ZIP 85041. Change my makeup kit to dark Brand A."}, config)
        result = await agent.run_turn("payment", {"kind": "confirmation", "accepted": True,
                                                   "proposal_id": proposal["request"]["proposal_id"]}, config)
        assert "charged to your" not in result["text"]
        assert "Credit card ending in 6918" in result["text"]
        assert "$10.36" in result["text"]
        assert "No real payment was sent" in result["text"]
        assert result["usage"]["original_model_answer"] == draft
    asyncio.run(run())
