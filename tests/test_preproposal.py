import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage

from test_agent_loop import call, runtime


def test_address_lookup_shows_owned_candidate_from_city_and_product(tmp_path):
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Lucas", "last_name": "Santos", "zip": "80239"}, "identity"),
        call("get_user_details", {"user_id": "lucas_santos_6600"}, "profile"),
        AIMessage(content="I can read the order for more detail."),
    ])
    result = asyncio.run(agent.run_turn("address", {"kind": "message", "text":
        "I'm Lucas Santos ZIP 80239. Tell me the shipping address for my daughter in Chicago and check the tablet."}, config))
    assert result["status"] == "answered"
    data = next(json.loads(m.content)["data"] for m in model.inputs[-1]
                if isinstance(m, ToolMessage) and m.name == "get_user_details")
    assert data["address_order_candidates"] == [
        {"order_id": "#W7895761", "city": "Chicago", "matched_terms": ["chicago", "tablet"]},
    ]


def test_preserve_material_and_color_rejects_partial_match_before_proposal(tmp_path):
    from retailflow.upstream.paths import RETAIL_DB_PATH
    raw = json.loads(RETAIL_DB_PATH.read_text())
    order = raw["orders"]["#W8835847"]
    user = raw["users"][order["user_id"]]
    args = {"order_id": order["order_id"], "item_ids": ["8895454203"],
            "new_item_ids": ["8920458606"], "payment_method_id": "gift_card_2652153"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": user["email"]}, "identity"),
        call("modify_pending_order_items", args, "wrong-variant"),
        AIMessage(content="I could not find an eligible replacement."),
    ])
    result = asyncio.run(agent.run_turn("preserve", {"kind": "message", "text":
        f"My email is {user['email']}. Change my bookshelf to 4 feet tall, keeping its material and color. "
        "If unavailable, cancel the whole order."}, config))
    assert result["status"] == "answered"
    assert result["usage"]["tool_attempts"][-1]["code"] == "INVALID_ARGUMENT"
    assert agent.environment.pending_proposal("preserve") is None


def test_cannot_claim_pending_order_was_cancelled_without_a_write(tmp_path):
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        AIMessage(content='**Order #W5056519 (Makeup Kit)** canceled — reason: "ordered by mistake".'),
    ])
    result = asyncio.run(agent.run_turn("claim", {"kind": "message", "text":
        "I'm Yara Muller ZIP 85041. Cancel #W5056519 because I ordered it by mistake."}, config))
    assert result["status"] == "failed"
    assert result["failure_reason"] == "unverified_cancellation_claim"
    assert agent.environment.get_order("claim", "#W5056519")["status"] == "pending"


def test_address_source_takes_precedence_over_item_and_target_city():
    from retailflow.preproposal import address_order_candidates

    business = {"orders": {
        "#W1": {"user_id": "owner", "address": {"city": "Los Angeles"}, "items": [{"name": "Bluetooth Speaker"}]},
        "#W2": {"user_id": "owner", "address": {"city": "New York"}, "items": [{"name": "Luggage Set"}]},
    }}

    def candidates(text):
        return [row["order_id"] for row in address_order_candidates(text, business, "owner", ["#W1", "#W2"])]

    assert candidates("Change the Bluetooth speaker in my LA order. Copy my NYC address from my other order.") == ["#W2"]
    assert candidates("Copy the address from order #W2 to order #W1.") == ["#W2"]
    assert candidates("Copy the address from my luggage order to the Bluetooth speaker order.") == ["#W2"]
    assert candidates("Change the Bluetooth speaker and change its address.") == []
    business["orders"]["#W2"]["user_id"] = "other"
    assert candidates("Copy the address from order #W2 to order #W1.") == []


def test_keep_clause_only_checks_the_named_product():
    from retailflow.preproposal import preserved_variant_conflicts

    business = {
        "orders": {"#W1": {"user_id": "owner", "items": [
            {"item_id": "old-shirt", "name": "T-Shirt", "options": {"color": "blue"}, "product_id": "shirt"},
            {"item_id": "old-shelf", "name": "Bookshelf", "options": {"color": "white"}, "product_id": "shelf"},
        ]}},
        "products": {"shirt": {"variants": {"new-shirt": {"options": {"color": "red"}}}}},
    }
    args = {"order_id": "#W1", "item_ids": ["old-shirt"], "new_item_ids": ["new-shirt"]}
    assert preserved_variant_conflicts(
        "In order #W2, change my T-shirt to size M, keeping its color. In order #W1, change my T-shirt to red.",
        business, "owner", args) == []
    assert preserved_variant_conflicts(
        "Change the T-shirt to red and change the bookshelf to 4 feet, keeping its color.", business, "owner", args) == []


def test_cancellation_claim_distinguishes_operation_from_historical_status():
    from retailflow.preproposal import false_cancellation_claim

    business = {"orders": {
        "#W1": {"user_id": "owner", "status": "pending"},
        "#W2": {"user_id": "owner", "status": "cancelled"},
    }}
    assert not false_cancellation_claim("You asked whether order #W1 was canceled. It is pending.", business, "owner")
    assert not false_cancellation_claim("I did not get #W1 canceled.", business, "owner")
    assert false_cancellation_claim("Order #W1 has now been canceled.", business, "owner")
    assert false_cancellation_claim("- **Order #W1** (jacket) — canceled by mistake.", business, "owner")
    assert false_cancellation_claim("I have canceled order #W1.", business, "owner")
    assert not false_cancellation_claim("Order #W2 was canceled.", business, "owner")
    assert false_cancellation_claim("Order #W2 has now been canceled for you.", business, "owner")
    assert false_cancellation_claim("I just canceled order #W2.", business, "owner")
    assert not false_cancellation_claim("I just canceled order #W2.", business, "owner", {"#W2"})


def test_keep_clause_survives_a_bare_zip_clarification():
    from retailflow.address_followups import advisory_opening

    opening = "Change my bookshelf to 4 feet tall, keeping its material and color."
    assert advisory_opening([opening, "94102"]) == opening
    assert advisory_opening([opening, "My name is Lucas Santos, ZIP 80239. Please look for the matching order in my account."]) == opening
    assert advisory_opening([opening, "You name is Daiki Silva and your email is daiki@example.com."]) == opening
    assert advisory_opening([opening, "Actually, wood is fine."]) == ""
