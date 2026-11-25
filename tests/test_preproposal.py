import asyncio
import json

import pytest

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
    assert false_cancellation_claim(
        "As a fallback, I have canceled the entire order. Your order #W1 has been canceled.",
        business, "owner",
    )
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


def test_copy_to_old_address_keeps_source_and_destination_separate(tmp_path):
    email = "sophia.martin4832@example.com"
    user_id = "sophia_martin_8570"
    source, target = "#W1092119", "#W1603792"
    address = {"address1": "592 Elm Avenue", "address2": "Suite 978", "city": "Houston",
               "state": "TX", "country": "USA", "zip": "77242"}
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_user_details", {"user_id": user_id}, "profile"),
        call("modify_pending_order_address", {"order_id": source, **address}, "wrong"),
        call("modify_pending_order_address", {"order_id": target, **{key: value for key, value in address.items()
                                                                     if key != "country"}}, "missing-country"),
        call("modify_pending_order_address", {"order_id": target, **address}, "right"),
    ])
    result = asyncio.run(agent.run_turn("copy", {"kind": "message", "text":
        f"My email is {email}. Copy the shipping address from my luggage order to the order "
        "still going to my old house and set it as my account default."}, config))
    assert result["request"]["args"]["order_id"] == target
    assert [row["code"] for row in result["usage"]["tool_attempts"]][-3:] == [
        "INVALID_ARGUMENT", "INVALID_ARGUMENT", "CONFIRMATION_REQUIRED",
    ]
    profile = next(json.loads(m.content)["data"] for m in model.inputs[-1]
                   if isinstance(m, ToolMessage) and m.name == "get_user_details")
    assert profile["address_copy"]["source_order_id"] == source
    assert profile["address_copy"]["destination_order_id"] == target
    assert agent.environment.get_order("copy", target)["address"] != address


def test_duplicate_product_proposal_targets_old_option_value_after_clarification(tmp_path):
    user_id, email = "aarav_santos_2259", "aarav.santos8320@example.com"
    order_id = "#W9672333"
    original, other, replacement = "1684786391", "3778566150", "5052031638"
    args = {"order_id": order_id, "new_item_ids": [replacement], "payment_method_id": "paypal_7664977"}
    agent, model, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity", "order"], "question": "Can I get your email and order number?"}, "ask"),
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_user_details", {"user_id": user_id}, "profile"),
        call("get_order_details", {"order_id": order_id}, "read"),
        call("modify_pending_order_items", {**args, "item_ids": ["3714494375"],
                                           "new_item_ids": ["3709608322"]}, "wrong-product"),
        call("modify_pending_order_items", {**args, "item_ids": [other]}, "wrong"),
        call("modify_pending_order_items", {**args, "item_ids": [original],
                                           "new_item_ids": ["2216662955"]}, "wrong-size"),
        call("modify_pending_order_items", {**args, "item_ids": [original]}, "right"),
    ], max_model_calls_per_turn=8)

    async def run():
        first = await agent.run_turn("item", {"kind": "message", "text":
            "Modify the 17-inch laptop in another order to 13-inch."}, config)
        assert first["status"] == "needs_input"
        return await agent.run_turn("item", {"kind": "message", "text":
            f"You are {user_id} and aarav.santos8321@example.com and {email}. "
            "I don't remember; please look through my orders."}, config)

    result = asyncio.run(run())
    assert result["request"]["args"]["item_ids"] == [original]
    assert result["request"]["args"]["new_item_ids"] == [replacement]
    assert [row["code"] for row in result["usage"]["tool_attempts"]][-4:] == [
        "INVALID_ARGUMENT", "INVALID_ARGUMENT", "INVALID_ARGUMENT", "CONFIRMATION_REQUIRED",
    ]
    order = next(json.loads(m.content)["data"] for m in model.inputs[-1]
                 if isinstance(m, ToolMessage) and m.name == "get_order_details")
    assert order["item_change_targets"][0]["item_id"] == original
    assert order["item_change_targets"][0]["options"]["screen size"] == "17-inch"


def test_copy_abstains_when_old_address_matches_multiple_pending_orders():
    from copy import deepcopy

    from retailflow.preproposal import address_copy_target
    from retailflow.upstream.paths import RETAIL_DB_PATH

    db = json.loads(RETAIL_DB_PATH.read_text())
    user_id, target = "sophia_martin_8570", "#W1603792"
    text = "Copy the shipping address from my luggage order to the order going to my old address."
    assert address_copy_target(text, db, user_id)["destination_order_id"] == target
    db["orders"]["#W1"] = {**deepcopy(db["orders"][target]), "order_id": "#W1"}
    db["users"][user_id]["orders"].append("#W1")
    assert address_copy_target(text, db, user_id) is None


def test_shipping_copy_does_not_stage_an_unrequested_account_change(tmp_path):
    email, user_id = "sophia.martin4832@example.com", "sophia_martin_8570"
    address = {"address1": "592 Elm Avenue", "address2": "Suite 978", "city": "Houston",
               "state": "TX", "country": "USA", "zip": "77242"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("modify_user_address", {"user_id": user_id, **address}, "wrong"),
        call("modify_pending_order_address", {"order_id": "#W1603792", **address}, "right"),
    ])
    result = asyncio.run(agent.run_turn("shipping-only", {"kind": "message", "text":
        f"My email is {email}. Copy the shipping address from my luggage order to the order "
        "still going to my old house."}, config))
    assert result["request"]["name"] == "modify_pending_order_address"
    assert result["usage"]["tool_attempts"][-2]["code"] == "INVALID_ARGUMENT"


def test_copy_evidence_survives_account_commit_without_a_profile_read(tmp_path):
    email, user_id = "sophia.martin4832@example.com", "sophia_martin_8570"
    source, target = "#W1092119", "#W1603792"
    address = {"address1": "592 Elm Avenue", "address2": "Suite 978", "city": "Houston",
               "state": "TX", "country": "USA", "zip": "77242"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("modify_user_address", {"user_id": user_id, **address}, "default"),
        call("modify_pending_order_address", {"order_id": source, **address, "address1": "12 Other St"}, "wrong"),
        call("modify_pending_order_address", {"order_id": target, **address}, "shipping"),
    ])

    async def run():
        first = await agent.run_turn("persist", {"kind": "message", "text":
            f"My email is {email}. Copy the shipping address from my luggage order to the order "
            "still going to my old house and set it as my account default."}, config)
        assert first["request"]["name"] == "modify_user_address"
        return await agent.run_turn("persist", {"kind": "confirmation", "accepted": True,
                                                "proposal_id": first["request"]["proposal_id"]}, config)

    result = asyncio.run(run())
    assert result["request"]["args"]["order_id"] == target
    assert result["usage"]["tool_attempts"][-2]["code"] == "INVALID_ARGUMENT"
    assert agent.environment.get_order("persist", source)["address"] == address


def test_copy_abstains_if_a_separate_address_change_is_explicit():
    from retailflow.preproposal import address_copy_target
    from retailflow.upstream.paths import RETAIL_DB_PATH

    db = json.loads(RETAIL_DB_PATH.read_text())
    text = ("Copy the shipping address from my luggage order to the order going to my old address. "
            "Also change the address for order #W1092119 to 129 Main St.")
    assert address_copy_target(text, db, "sophia_martin_8570") is None


def test_item_hint_abstains_on_a_conflicting_another_order_reference():
    from retailflow.preproposal import item_change_targets
    from retailflow.upstream.paths import RETAIL_DB_PATH

    db = json.loads(RETAIL_DB_PATH.read_text())
    text = "Check order #W9672333. Modify the 17-inch laptop in another order to 13-inch."
    assert item_change_targets(text, db, "aarav_santos_2259") == []



def test_cheapest_available_variant_is_checked_before_proposing(tmp_path):
    email = "sophia.martin4832@example.com"
    args = {"order_id": "#W1603792", "item_ids": ["6501071631"], "payment_method_id": "credit_card_5694100"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_order_details", {"order_id": args["order_id"]}, "order"),
        call("get_product_details", {"product_id": "8024098596"}, "catalog"),
        call("modify_pending_order_items", {**args, "new_item_ids": ["4615543240"]}, "wrong"),
        call("modify_pending_order_items", {**args, "new_item_ids": ["2106335193"]}, "right"),
    ])
    result = asyncio.run(agent.run_turn("cheapest", {"kind": "message", "text":
        f"My email is {email}. Change my pending tablet to the cheapest available version."}, config))
    assert result["request"]["args"]["new_item_ids"] == ["2106335193"]
    assert result["usage"]["tool_attempts"][-2]["code"] == "INVALID_ARGUMENT"


def test_another_order_cannot_propose_a_change_to_the_referenced_order(tmp_path):
    email = "aarav.santos8320@example.com"
    order_id = "#W9672333"
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("modify_pending_order_items", {"order_id": order_id, "item_ids": ["1684786391"],
                                           "new_item_ids": ["5052031638"],
                                           "payment_method_id": "paypal_7664977"}, "wrong"),
        AIMessage(content="I could not verify the other order."),
    ])
    result = asyncio.run(agent.run_turn("another", {"kind": "message", "text":
        f"My email is {email}. Check order {order_id}. Modify the 17-inch laptop in another order to 13-inch."},
        config))
    assert result["status"] == "answered"
    assert result["usage"]["tool_attempts"][-1]["code"] == "INVALID_ARGUMENT"
    assert agent.environment.pending_proposal("another") is None


def test_confirming_tablet_change_leaves_two_separately_confirmed_addresses(tmp_path):
    email, user_id = "sophia.martin4832@example.com", "sophia_martin_8570"
    order_id = "#W1603792"
    address = {"address1": "592 Elm Avenue", "address2": "Suite 978", "city": "Houston",
               "state": "TX", "country": "USA", "zip": "77242"}
    tablet = {"order_id": order_id, "item_ids": ["6501071631"], "new_item_ids": ["2106335193"],
              "payment_method_id": "credit_card_5694100"}
    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_user_details", {"user_id": user_id}, "profile"),
        call("modify_pending_order_items", tablet, "tablet"),
        call("modify_pending_order_items", {**tablet, "new_item_ids": ["2106335199"]}, "repeat-tablet"),
        call("modify_pending_order_address", {"order_id": order_id, **address}, "shipping"),
        call("modify_user_address", {"user_id": user_id, **address}, "default"),
        AIMessage(content="Both addresses were updated."),
        AIMessage(content="Both addresses were updated."),
    ])

    async def run():
        message = {"kind": "message", "text": f"My email is {email}. Copy the shipping address from my luggage order "
                   "to the order still going to my old house and set it as my account default. "
                   "Change my pending tablet to the cheapest available version."}
        first = await agent.run_turn("three-writes", message, config)
        assert first["request"]["name"] == "modify_pending_order_items"

        async def confirm(proposal):
            return await agent.run_turn("three-writes", {"kind": "confirmation", "accepted": True,
                                                         "proposal_id": proposal["request"]["proposal_id"]}, config)

        shipping = await confirm(first)
        assert shipping["request"]["name"] == "modify_pending_order_address"
        assert shipping["usage"]["tool_attempts"][-2]["code"] == "INVALID_ARGUMENT"
        assert agent.environment.get_order("three-writes", order_id)["address"] != address
        profile = await confirm(shipping)
        assert profile["request"]["name"] == "modify_user_address"
        assert agent.environment.load_session("three-writes")["business"]["users"][user_id]["address"] != address
        final = await confirm(profile)
        assert final["status"] == "answered"
        assert agent.environment.get_order("three-writes", order_id)["address"] == address
        assert agent.environment.load_session("three-writes")["business"]["users"][user_id]["address"] == address
        assert len({p["request"]["proposal_id"] for p in (first, shipping, profile)}) == 3

    asyncio.run(run())
    assert not model.responses


def test_item_lookup_promise_after_authentication_gets_one_read_retry(tmp_path):
    email, user_id = "aarav.santos8320@example.com", "aarav_santos_2259"
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        AIMessage(content="I will now retrieve your order history to locate the laptop. Please wait while I search."),
        call("get_user_details", {"user_id": user_id}, "profile"),
        AIMessage(content="Your laptop is in a pending order."),
    ])
    result = asyncio.run(agent.run_turn("read-retry", {"kind": "message", "text":
        f"My email is {email}. Modify the 17-inch laptop to 13-inch."}, config))
    assert [row["name"] for row in result["usage"]["tool_attempts"]] == [
        "find_user_id_by_email", "get_user_details",
    ]
    assert result["status"] == "answered"


def test_read_retry_does_not_consume_the_structured_clarification_retry(tmp_path):
    email, user_id = "aarav.santos8320@example.com", "aarav_santos_2259"
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        AIMessage(content="I will now retrieve your order history to locate the laptop."),
        call("get_user_details", {"user_id": user_id}, "profile"),
        AIMessage(content="Please provide your preferred payment method."),
        call("ask_user", {"fields": ["payment_method"], "question": "What payment method should I use?"}, "clarify"),
    ])
    result = asyncio.run(agent.run_turn("read-then-clarify", {"kind": "message", "text":
        f"My email is {email}. Modify the 17-inch laptop to 13-inch."}, config))
    assert result["status"] == "needs_input"
    assert result["request"]["fields"] == ["payment_method"]


@pytest.mark.parametrize("keep_order", [False, True])
def test_cross_product_fallback_uses_the_cancel_reason_and_a_separate_confirmation(tmp_path, keep_order):
    order_id = "#W3361211"
    agent, _, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity"], "question": "What is your name and ZIP?"}, "identity-question"),
        call("find_user_id_by_name_zip", {"first_name": "Aarav", "last_name": "Lee", "zip": "85025"}, "identity"),
        call("get_order_details", {"order_id": order_id}, "order"),
        AIMessage(content="I will keep the order." if keep_order else
                  f"Order {order_id} was canceled. The order is currently pending."),
    ])

    async def run():
        thread = "product-fallback"
        opening = "Change my luggage set to a coat. If that cannot be done, return that item; if that is also impossible, cancel the whole order."
        first = await agent.run_turn(thread, {"kind": "message", "text": opening}, config)
        assert first["request"]["fields"] == ["identity"]
        reason = await agent.run_turn(thread, {"kind": "message", "text":
            "You are Aarav Lee. You live in Phoenix, AZ 85025."}, config)
        assert reason["status"] == "needs_input"
        assert reason["request"]["fields"] == ["cancel_reason"]
        assert agent.environment.pending_proposal(thread) is None

        proposal = await agent.run_turn(thread, {"kind": "message",
                                                "text": "Keep it." if keep_order else "I no longer need it."}, config)
        if keep_order:
            assert proposal["status"] == "answered"
            assert agent.environment.pending_proposal(thread) is None
            assert agent.environment.get_order(thread, order_id)["status"] == "pending"
            return
        assert proposal["status"] == "needs_confirmation"
        assert proposal["request"]["name"] == "cancel_pending_order"
        assert proposal["request"]["args"] == {"order_id": order_id, "reason": "no longer needed"}
        assert agent.environment.get_order(thread, order_id)["status"] == "pending"
        assert agent.environment.receipt_for_proposal(thread, proposal["request"]["proposal_id"]) is None

        final = await agent.run_turn(thread, {"kind": "confirmation", "accepted": True,
                                             "proposal_id": proposal["request"]["proposal_id"]}, config)
        assert final["status"] == "answered"
        assert f"Order {order_id} was cancelled" in final["text"]
        assert "currently pending" not in final["text"]
        assert "no longer needed" in final["text"]
        assert "different product was unavailable" in final["text"]
        assert "ineligible for return before cancellation" in final["text"]
        assert "$1464.00" in final["text"]
        assert "5 to 7 business days" in final["text"]
        assert "No real refund was issued" in final["text"]
        assert "currently pending" in final["usage"]["original_model_answer"]
        assert agent.environment.get_order(thread, order_id)["status"] == "cancelled"
        assert agent.environment.get_order(thread, "#W3586556")["status"] == "pending"

    asyncio.run(run())


def test_cross_product_cancel_target_abstains_for_delivered_ambiguous_or_variant_requests():
    from retailflow.preproposal import cross_product_cancel_target
    from retailflow.upstream.paths import RETAIL_DB_PATH

    business = json.loads(RETAIL_DB_PATH.read_text())
    user_id, order_id = "aarav_lee_1982", "#W3361211"
    chain = ". If that cannot be done, return that item; if that is also impossible, cancel the whole order."
    text = "Change my luggage set to a coat" + chain
    assert cross_product_cancel_target(text, business, user_id) == order_id
    assert cross_product_cancel_target("Change my luggage set to a blue one" + chain, business, user_id) is None
    assert cross_product_cancel_target(text, business, "yara_muller_8652") == "#W5995614"
    business["users"]["yara_muller_8652"]["orders"] = [order_id]
    assert cross_product_cancel_target(text, business, "yara_muller_8652") is None
    business["orders"][order_id]["status"] = "delivered"
    assert cross_product_cancel_target(text, business, user_id) is None
    business["orders"][order_id]["status"] = "pending"
    business["orders"]["#W1"] = {**business["orders"][order_id], "order_id": "#W1"}
    business["users"][user_id]["orders"].append("#W1")
    assert cross_product_cancel_target(text, business, user_id) is None
