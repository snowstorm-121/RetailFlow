from retailflow.address_followups import explicit_address_steps, matching_variants, processing_guarantee_fact


def test_explicit_address_steps_distinguish_copy_source_and_write_targets():
    steps = explicit_address_steps(
        "Copy the shipping address from order #W8277957 to #W5056519. "
        "In #W5056519 change the light professional Brand B makeup kit to a dark Brand A kit. "
        "Also cancel #W5995614 because I ordered it by mistake."
    )
    assert [(s["tool"], s["order_id"]) for s in steps] == [
        ("modify_pending_order_address", "#W5056519"),
        ("modify_pending_order_items", "#W5056519"),
        ("cancel_pending_order", "#W5995614"),
    ]
    assert steps[1]["preferences"] == "a dark Brand A kit"


def test_processing_condition_provides_fact_but_does_not_infer_older_order():
    text = ("Compare #W2702727 and #W8268610. If you cannot guarantee the older one "
            "will be processed within five days, cancel it.")
    assert explicit_address_steps(text) == []
    assert "cannot guarantee" in processing_guarantee_fact(text)
    assert "ask which order" in processing_guarantee_fact(text)


def test_catalog_matches_only_available_variants_using_requested_option_values():
    variants = {
        "1": {"available": True, "options": {"skin tone": "dark", "brand": "Brand A"}, "price": 10},
        "2": {"available": False, "options": {"skin tone": "dark", "brand": "Brand A"}, "price": 9},
        "3": {"available": True, "options": {"skin tone": "light", "brand": "Brand A"}, "price": 8},
        "4": {"available": True, "options": {"skin tone": "dark", "brand": "Brand B"}, "price": 7},
    }
    assert matching_variants(variants, "a dark Brand A kit") == [
        {"item_id": "1", "options": variants["1"]["options"], "price": 10, "matched_options": {"skin tone": "dark", "brand": "Brand A"}},
    ]
    assert matching_variants(variants, "not Brand A, use Brand B") == []
    assert matching_variants(variants, "cheapest variant") == []


def test_owned_order_catalog_hints_reach_the_model_without_writing(tmp_path):
    import asyncio
    import json

    from langchain_core.messages import AIMessage, ToolMessage

    from test_agent_loop import call, runtime

    agent, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        call("get_order_details", {"order_id": "#W5056519"}, "order"),
        call("get_product_details", {"product_id": "5149340237"}, "catalog"),
        AIMessage(content="I found an available candidate."),
    ])
    order = "#W5056519"
    agent.environment.create_session("catalog", config["initial_state_ref"])
    baseline = agent.environment.state_hash("catalog")
    result = asyncio.run(agent.run_turn("catalog", {"kind": "message", "text":
        "I'm Yara Muller ZIP 85041. Copy my shipping address from #W8277957 to #W5056519. "
        "In #W5056519 change the light Brand B makeup kit to a dark Brand A kit. "
        "Also cancel #W5995614."}, config))
    assert result["status"] == "answered"
    catalog = next(json.loads(message.content)["data"] for message in model.inputs[-1]
                   if isinstance(message, ToolMessage) and message.name == "get_product_details")
    assert [row["item_id"] for row in catalog["available_option_hits"]] == ["1573035764"]
    assert len(catalog["variants"]) > 1
    assert order in next(message.content for message in model.inputs[-1] if message.type == "system")
    assert agent.environment.state_hash("catalog") == baseline


def test_negated_and_conditional_commands_are_not_checklist_entries():
    for cancellation in (
        "Do not cancel #W5995614.",
        "Only if the item is out of stock, cancel #W5995614.",
        "Cancel #W5995614 only if the item is out of stock.",
        "Cancel #W5995614 or #W8277957, whichever is older.",
        "Cancel #W5995614 only after it ships.",
    ):
        steps = explicit_address_steps(cancellation + " Change the shipping address for #W5056519.")
        assert [step["tool"] for step in steps] == ["modify_pending_order_address"]


def test_later_customer_correction_invalidates_opening_hints():
    from retailflow.address_followups import advisory_opening
    opening = "In #W5056519 change the makeup kit to dark Brand A. Change shipping."
    assert advisory_opening([opening, "My email is yara@example.com.", "I explicitly confirm the displayed proposal."]) == opening
    assert advisory_opening([opening, "My email is yara@example.com. Use Brand B."]) == ""
    assert advisory_opening([opening, "Brand B."]) == ""


def test_catalog_hits_do_not_claim_to_match_an_unknown_requested_option():
    variants = {
        "1": {"available": True, "options": {"color": "blue", "voltage": "220V"}, "price": 10},
    }
    hit, = matching_variants(variants, "a blue 110V lamp")
    assert hit["matched_options"] == {"color": "blue"}
    assert hit["options"]["voltage"] == "220V"


def test_alternative_shipping_destinations_are_not_checklist_entries():
    assert explicit_address_steps(
        "Copy the shipping address from #W8277957 to #W5056519 or #W5995614, whichever is pending."
    ) == []


def test_promised_shipping_proposal_recovers_as_native_tool_call(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import call, runtime

    args = {"order_id": "#W8665881", "address1": "123 Elm Street", "address2": "Suite 641",
            "city": "Austin", "state": "TX", "country": "USA", "zip": "78712"}
    agent, _, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Fatima", "last_name": "Johnson", "zip": "78712"}, "identity"),
        call("get_order_details", {"order_id": args["order_id"]}, "read"),
        AIMessage(content="Shipping address change for this order. I will now propose this change. Would you like to proceed with confirming the address update?"),
        call("modify_pending_order_address", args, "address"),
    ], isolated_proposal_repair=True)
    result = asyncio.run(agent.run_turn("address", {"kind": "message", "text":
        "I'm Fatima Johnson ZIP 78712. Change shipping for #W8665881 to Suite 641."}, config))
    assert result["status"] == "needs_confirmation"
    assert agent.environment.get_order("address", args["order_id"])["address"]["address2"] == "Suite 640"
