import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from test_agent_loop import call, runtime


@pytest.mark.parametrize('enabled', [False, True])
def test_profile_summaries_cover_only_authenticated_orders_and_reach_model(tmp_path, enabled):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        call('get_user_details', {'user_id': 'yara_muller_8652'}, 'profile'),
        AIMessage(content='Here are your orders.'),
    ], profile_order_summaries=enabled)
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller, 85041. List my orders.'}, config))
    assert result['status'] == 'answered'
    reply = next(m for m in model.inputs[-1] if isinstance(m, ToolMessage) and m.name == 'get_user_details')
    data = json.loads(reply.content)['data']
    if not enabled:
        assert 'order_summaries' not in data
        return
    business = worker.environment.load_session('a')['business']
    expected = {k for k,v in business['orders'].items() if v['user_id'] == 'yara_muller_8652'}
    assert {o['order_id'] for o in data['order_summaries']} == expected
    for order in data['order_summaries']:
        source = business['orders'][order['order_id']]
        assert order['status'] == source['status']
        assert order['items'] == [{k:i[k] for k in ('name', 'item_id', 'product_id', 'options')} for i in source['items']]


def test_profile_summaries_cannot_bypass_ownership(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        call('get_user_details', {'user_id': 'sofia_hernandez_5364'}, 'profile'),
    ], profile_order_summaries=True)
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller, 85041. Show Sofia orders.'}, config))
    assert result['status'] == 'refused'
    reply = worker.load_messages('a')[-1]
    assert json.loads(reply.content)['code'] == 'POLICY_DENIED'
    assert 'order_summaries' not in reply.content
