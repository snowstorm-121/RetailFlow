import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from test_agent_loop import call, runtime


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('product_id', ['9523456873', 'not-a-product'])
def test_catalog_count_survives_context_and_return_receipt(tmp_path, enabled, product_id):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_product_details', {'product_id': product_id}, 'catalog'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
        AIMessage(content='Return requested. There are 999 options.'),
    ], return_receipt_response=True, catalog_availability_facts=enabled)
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. How many T-shirt options are available? Return my headphones to original payment.'}, config))
    assert first['status'] == 'needs_confirmation'
    views = [m for m in model.inputs[-1] if isinstance(m, ToolMessage) and m.name == 'get_product_details']
    data = json.loads(views[-1].content)
    valid = product_id == '9523456873'
    if enabled and valid:
        expected = sum(v['available'] for v in data['data']['variants'].values())
        assert data['data']['available_variant_count'] == expected
    else:
        assert 'available_variant_count' not in data.get('data', {})
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation', 'accepted': True,
        'proposal_id': first['request']['proposal_id']}, config))
    assert result['status'] == 'answered'
    assert '999' not in result['text']
    assert ('available variants' in result['text']) == (enabled and valid)
    if enabled and valid:
        assert f'T-Shirt (9523456873): {expected} available variants' in result['text']
    assert 'refund is not settled' in result['text']


def test_receipt_keeps_latest_successful_lookup_for_each_product(tmp_path):
    from retailflow.upstream.paths import RETAIL_DB_PATH
    products = json.loads(RETAIL_DB_PATH.read_text())['products']
    other_id = next(key for key in products if key != '9523456873')
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_product_details', {'product_id': '9523456873'}, 'first'),
        call('get_product_details', {'product_id': other_id}, 'other'),
        call('get_product_details', {'product_id': '9523456873'}, 'second'),
        call('get_product_details', {'product_id': '9523456873'}, 'failed'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
        AIMessage(content='Done.'),
    ], return_receipt_response=True, catalog_availability_facts=True, max_model_calls_per_turn=12)
    dispatch = worker.environment.dispatch_tool
    calls = 0

    def changed_catalog(thread, name, args):
        nonlocal calls
        if name == 'get_product_details' and args['product_id'] == '9523456873':
            calls += 1
            if calls == 3:
                return {'ok': False, 'code': 'TOOL_ERROR', 'message': 'Lookup failed.'}
            result = dispatch(thread, name, args)
            if calls == 2:
                for variant in result['data']['variants'].values():
                    variant['available'] = False
            return result
        return dispatch(thread, name, args)

    worker.environment.dispatch_tool = changed_catalog
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Check products and return headphones to original payment.'}, config))
    assert first['status'] == 'needs_confirmation'
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation', 'accepted': True,
        'proposal_id': first['request']['proposal_id']}, config))
    assert result['status'] == 'answered'
    assert result['text'].count('T-Shirt (9523456873):') == 1
    assert 'T-Shirt (9523456873): 0 available variants' in result['text']
    other_count = sum(v['available'] for v in products[other_id]['variants'].values())
    assert f"{products[other_id]['name']} ({other_id}): {other_count} available variants" in result['text']



@pytest.mark.parametrize("catalog_changed", [False, True])
def test_catalog_exposes_verified_preference_matches_after_return_confirmation(tmp_path, catalog_changed):
    email = "aarav.santos8320@example.com"
    speaker_order = "#W8528674"
    worker, model, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_product_details", {"product_id": "4760268021"}, "catalog"),
        call("get_order_details", {"order_id": speaker_order}, "speaker"),
        call("return_delivered_order_items", {"order_id": speaker_order, "item_ids": ["6704763132"],
                                             "payment_method_id": "paypal_7664977"}, "return"),
        call("ask_user", {"fields": ["payment_method"], "question": "Which payment method for the item change?"}, "method"),
    ])

    async def run():
        thread = "preferred"
        first = await worker.run_turn(thread, {"kind": "message", "text":
            f"My email is {email}. Return the more expensive speaker that is not water resistant. "
            "Modify the 17-inch laptop in another order to 13-inch. If the exact version is unavailable, "
            "list all available 13-inch options; I prefer i5 over i7 and silver or black over other colors."}, config)
        assert first["status"] == "needs_confirmation"
        raw = next(m for m in worker.load_messages(thread) if isinstance(m, ToolMessage) and m.tool_call_id == "catalog")
        view = next(m for m in model.inputs[2] if isinstance(m, ToolMessage) and m.tool_call_id == "catalog")
        raw_data, visible = [json.loads(m.content)["data"] for m in (raw, view)]
        assert raw_data["variants"]["3778566150"]["available"] is False
        assert set(row["item_id"] for row in visible["available_option_hits"]) == {"1657832319", "5052031638", "6056040996"}
        assert visible["variants"]["3778566150"]["available"] is False
        assert [v["item_id"] for v in visible["preferred_available_option_hits"]] == ["5052031638"]
        assert "5052031638" in model.inputs[2][-1].content
        if catalog_changed:
            business = worker.environment.load_session(thread)["business"]
            business["products"]["4760268021"]["variants"]["5052031638"]["available"] = False
            with worker.environment.storage.transaction() as db:
                db.execute("UPDATE sessions SET business_json = ? WHERE thread = ?", (json.dumps(business), thread))
        await worker.run_turn(thread, {"kind": "confirmation", "accepted": True,
                                      "proposal_id": first["request"]["proposal_id"]}, config)
        candidate = '"preferred_available_option_hits": [{"item_id": "5052031638"'
        assert (candidate in model.inputs[-1][0].content) is not catalog_changed
        if catalog_changed:
            assert not any(candidate in message.content for message in model.inputs[-1])
        assert worker.environment.pending_proposal(thread) is None

    asyncio.run(run())


def test_catalog_cheapest_summary_excludes_unavailable_and_retains_ties(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call("find_user_id_by_name_zip", {"first_name": "Mei", "last_name": "Davis", "zip": "80217"}, "identity"),
        call("get_product_details", {"product_id": "2747247837"}, "catalog"),
        AIMessage(content="I have read the catalog."),
    ])
    worker.environment.create_session("cheapest", config["initial_state_ref"])
    business = worker.environment.load_session("cheapest")["business"]
    business["products"]["2747247837"]["variants"] = {
        key: {"item_id": key, "options": {"color": key}, "price": price, "available": available}
        for key, price, available in [("red", 1, False), ("green", 12, True), ("blue", 8, True), ("black", 8, True)]
    }
    with worker.environment.storage.transaction() as db:
        db.execute("UPDATE sessions SET business_json = ? WHERE thread = ?", (json.dumps(business), "cheapest"))
    result = asyncio.run(worker.run_turn("cheapest", {"kind": "message",
        "text": "Mei Davis ZIP 80217. Can I exchange the pet bed for the cheapest version?"}, config))
    assert result["status"] == "answered"
    read = next(m for m in model.inputs[-1] if isinstance(m, ToolMessage) and m.tool_call_id == "catalog")
    cheapest = json.loads(read.content)["data"]["cheapest_available_variants"]
    assert {(variant["item_id"], variant["price"]) for variant in cheapest} == {("blue", 8), ("black", 8)}
