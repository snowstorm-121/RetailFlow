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

