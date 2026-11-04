import asyncio

import pytest
from langchain_core.messages import AIMessage

from test_agent_loop import call, runtime


@pytest.mark.parametrize('session,writes', [
    ({}, set()),
    ({'user_id': 'u'}, set()),
    ({'user_id': 'u', 'target_order_status': 'pending'}, {'cancel_pending_order'}),
    ({'user_id': 'u', 'target_order_status': 'delivered'}, {'return_delivered_order_items'}),
    ({'user_id': 'u', 'target_order_status': 'cancelled'}, set()),
    ({'user_id': 'u', 'target_order_status': 'return requested'}, set()),
])
def test_tool_visibility_preserves_discovery_and_limits_writes(session, writes):
    from react_agent.tools import retail_tools_for_session

    names = {t['function']['name'] for t in retail_tools_for_session(session)}
    assert {'ask_user', 'transfer_to_human_agents'} <= names
    assert names & {'cancel_pending_order', 'return_delivered_order_items'} == writes
    if session.get('user_id'):
        assert {'get_user_details', 'get_order_details', 'get_product_details'} <= names
    else:
        assert names == {'ask_user', 'transfer_to_human_agents',
                         'find_user_id_by_email', 'find_user_id_by_name_zip'}


def test_visibility_refreshes_after_identity_and_order_tools(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'u'),
        call('get_order_details', {'order_id': '#W5056519'}, 'o'),
        AIMessage(content='Your order is pending.'),
    ], state_tool_filter=True)
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yara Muller, 85041. Check #W5056519.'}, config))
    offered = [set(entry['tool_names']) for entry in result['usage']['context_inputs']]
    assert 'get_order_details' not in offered[0]
    assert 'get_order_details' in offered[1]
    assert 'cancel_pending_order' not in offered[1]
    assert 'cancel_pending_order' in offered[2]
    assert 'return_delivered_order_items' not in offered[2]
    assert result['status'] == 'answered'
