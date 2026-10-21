import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict

from test_agent_loop import call, runtime


FIELDS = ['identity', 'order', 'cancel_reason', 'return_items', 'refund_method']


def test_ask_user_schema_declares_the_application_field_contract():
    from react_agent.tools import RETAIL_TOOLS

    tool = next(t['function'] for t in RETAIL_TOOLS if t['function']['name'] == 'ask_user')
    schema = tool['parameters']['properties']['fields']
    assert schema['items'].get('enum') == FIELDS
    assert schema.get('minItems') == 1


def test_unknown_ask_field_is_rejected_before_pausing(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('ask_user', {'fields': ['customer_id'], 'question': 'Your ID?'}, 'invalid'),
        call('ask_user', {'fields': ['identity'], 'question': 'Your name and ZIP?'}, 'valid'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Return an item.'}, config))
    assert result['status'] == 'needs_input'
    assert result['request']['fields'] == ['identity']
    replies = [m for m in worker.load_messages('a') if isinstance(m, ToolMessage)]
    assert json.loads(replies[0].content)['code'] == 'INVALID_ARGUMENT'
    assert json.loads(replies[1].content)['code'] == 'NEEDS_INPUT'


def test_native_protocol_examples_are_model_context_not_session_events(tmp_path):
    worker, model, config = runtime(tmp_path, [AIMessage(content='Hello.')])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Hello'}, config))
    examples = [m for m in model.inputs[0] if m.additional_kwargs.get('protocol_example')]
    assert len(examples) == 6
    assert sum(isinstance(m, AIMessage) and bool(m.tool_calls) for m in examples) == 2
    assert not any(m.additional_kwargs.get('protocol_example') for m in worker.load_messages('a'))
    assert result['usage']['tool_calls'] == 0
    assert result['usage']['model_calls'] == 1
    assert result['usage']['context_inputs'][0]['messages'] == messages_to_dict(model.inputs[0])


def test_plain_information_question_gets_one_native_repair_not_fake_execution(tmp_path):
    worker, model, config = runtime(tmp_path, [
        AIMessage(content='Could you provide your email or name and ZIP?'),
        call('ask_user', {'fields': ['identity'], 'question': 'Your identity?'}, 'native'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my order.'}, config))
    assert result['status'] == 'needs_input'
    assert result['usage']['model_calls'] == 2
    assert result['usage']['dialogue_repairs'] == 1
    assert [t['function']['name'] for t in model.bound_tools] == ['ask_user']
    assert any(m.additional_kwargs.get('protocol_feedback') for m in model.inputs[1])
    assert [m.tool_call_id for m in worker.load_messages('a') if isinstance(m, ToolMessage)] == ['native']


def test_failed_native_repair_stops_and_does_not_claim_pause(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='Could you provide your email?'),
        AIMessage(content='What is your email?'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my order.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'information_request_not_structured'
    assert result['usage']['model_calls'] == 2 and result['usage']['tool_calls'] == 0


def test_fabricated_identity_is_rejected_before_business_lookup(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'guess'),
        call('ask_user', {'fields': ['identity'], 'question': 'Who are you?'}, 'ask'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Please cancel my order.'}, config))
    assert result['status'] == 'needs_input'
    assert worker.load_session('a')['user_id'] is None
    replies = [m for m in worker.load_messages('a') if isinstance(m, ToolMessage)]
    assert json.loads(replies[0].content)['code'] == 'MISSING_USER_INFORMATION'


def test_narrated_single_native_call_is_not_a_final_answer(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='I will locate the account.', tool_calls=[{'name': 'find_user_id_by_name_zip',
            'args': {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'id': 'lookup'}]),
        AIMessage(content='Account located.'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller, 85041. Find my account.'}, config))
    assert result['status'] == 'answered' and result['text'] == 'Account located.'
    assert worker.load_session('a')['user_id'] == 'yara_muller_8652'
    assert result['usage']['protocol_errors'] == 0


def test_protocol_feedback_expires_after_the_next_native_call(tmp_path):
    worker, model, config = runtime(tmp_path, [
        AIMessage(content='Could you provide your email?'),
        call('ask_user', {'fields': ['identity'], 'question': 'Your identity?'}, 'ask'),
        AIMessage(content='Thanks for the details.'),
    ])
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my order.'}, config))
    assert first['status'] == 'needs_input'
    asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller 85041'}, config))
    assert not any(m.additional_kwargs.get('protocol_feedback') for m in model.inputs[-1])
    assert any(m.additional_kwargs.get('protocol_feedback') for m in worker.load_messages('a'))


def test_text_confirmation_is_repaired_into_a_proposal_not_an_approval(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        call('get_order_details', {'order_id': '#W5056519'}, 'order'),
        AIMessage(content='The order is pending. Please confirm if you want to proceed with cancellation.'),
        call('cancel_pending_order', {'order_id': '#W5056519', 'reason': 'ordered by mistake'}, 'proposal'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yara Muller, 85041. Cancel #W5056519, ordered by mistake.'}, config))
    assert result['status'] == 'needs_confirmation'
    assert result['request']['args']['reason'] == 'ordered by mistake'
    assert worker.environment.get_order('a', '#W5056519')['status'] == 'pending'
    assert result['usage']['dialogue_repairs'] == 1
    assert len(model.bound_tools) == 9


def test_empty_model_reply_is_not_reported_as_an_answer(tmp_path):
    worker, _, config = runtime(tmp_path, [AIMessage(content=''), AIMessage(content='')])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Hello.'}, config))
    assert result['status'] == 'failed' and result['failure_reason'] == 'empty_model_response'
    assert result['usage']['model_calls'] == 2 and result['usage']['tool_calls'] == 0


def test_native_repair_rejects_a_tool_not_offered_in_that_request(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='Could you provide the order ID?'),
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'wrong-tool'),
        call('ask_user', {'fields': ['order'], 'question': 'Your order ID?'}, 'ask'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller 85041. Cancel my order.'}, config))
    assert result['status'] == 'needs_input'
    replies = [m for m in worker.load_messages('a') if isinstance(m, ToolMessage)]
    assert json.loads(replies[0].content)['code'] == 'INVALID_ARGUMENT'
    assert worker.load_session('a')['user_id'] is None


def test_unknown_information_field_is_not_counted_as_schema_legal():
    from retailflow.evaluation import _tool_stats

    stats, _ = _tool_stats([call('ask_user', {'fields': ['customer_id'], 'question': 'Your ID?'}, 'bad')])
    assert stats['model_tool_requests'] == 1 and stats['legal_tool_requests'] == 0


def test_missing_payment_question_is_not_mistaken_for_write_confirmation(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='I cannot proceed with the return until you provide your payment method.'),
        call('ask_user', {'fields': ['refund_method'], 'question': 'Your refund method?'}, 'ask'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return an item from #W2378156.'}, config))
    assert result['status'] == 'needs_input'
    assert [t['function']['name'] for t in model.bound_tools] == ['ask_user']
    assert worker.environment.pending_proposal('a') is None


def test_may_i_have_email_is_an_information_request(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='May I have your email address?'),
        call('ask_user', {'fields': ['identity'], 'question': 'Your email?'}, 'ask'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my order.'}, config))
    assert result['status'] == 'needs_input' and result['usage']['model_calls'] == 2


def test_information_retry_is_focused_without_replaying_failed_answer_as_assistant(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        AIMessage(content='Please provide a cancellation reason.'),
        call('ask_user', {'fields': ['cancel_reason'], 'question': 'Your cancellation reason?'}, 'native'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller 85041. Cancel my purchase.'}, config))
    retry = model.inputs[2]
    assert result['status'] == 'needs_input'
    assert not any(isinstance(m, AIMessage) and m.content == 'Please provide a cancellation reason.' for m in retry)
    assert any('Please provide a cancellation reason.' in str(m.content) for m in retry)
    assert not any(m.additional_kwargs.get('protocol_retry_input') for m in worker.load_messages('a'))
    assert result['usage']['context_inputs'][2]['messages'] == messages_to_dict(retry)


def test_information_retry_cannot_finish_with_unrelated_plain_text(tmp_path):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='Please provide your email.'), AIMessage(content='Thanks, I can help.'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my purchase.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'information_request_not_structured'


def test_information_retry_rechecks_the_quoted_request_budget(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        AIMessage(content='Please provide your payment method. ' + 'Details. ' * 1000),
        call('ask_user', {'fields': ['refund_method'], 'question': 'Your refund method?'}, 'should-not-run'),
    ])
    config['context_budget_tokens'] = 500
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Yara Muller 85041. Return an item.'}, config))
    assert result['status'] == 'budget_exceeded'
    assert result['failure_reason'] == 'context_budget'
    assert result['usage']['model_calls'] == 2
    assert worker._load_runtime('a')['model_calls'] == 2


def test_proceed_confirmation_with_known_refund_method_requires_business_proposal(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='Please confirm if you would like me to proceed with the return using your original payment method.'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the vacuum cleaner in #W2378156 to the original payment method.'}, config))
    assert result['status'] == 'needs_confirmation'
    assert len(model.bound_tools) == 9
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'


def test_direct_return_confirmation_cannot_finish_without_a_proposal(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='Would you like me to return these items?'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the vacuum cleaner in #W2378156 to the original payment method.'}, config))
    assert result['status'] == 'needs_confirmation'
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'


def test_confirmed_gift_card_receipt_exposes_actual_refund_facts(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Emma', 'last_name': 'Smith', 'zip': '10192'}, 'identity'),
        call('get_order_details', {'order_id': '#W2417020'}, 'order'),
        call('cancel_pending_order', {'order_id': '#W2417020', 'reason': 'no longer needed'}, 'proposal'),
        AIMessage(content='The order was cancelled.'),
    ])
    proposal = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Emma Smith, 10192. Cancel #W2417020; I no longer need it.'}, config))
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation',
        'proposal_id': proposal['request']['proposal_id'], 'accepted': True}, config))
    from langchain_core.messages import SystemMessage
    outcome = next(json.loads(m.content.removeprefix('Application outcome: ')) for m in model.inputs[-1]
                   if isinstance(m, SystemMessage) and m.additional_kwargs.get('application_result'))
    assert result['status'] == 'answered'
    assert outcome['refunds'][0]['payment_method_id'] == 'gift_card_8541487'
    assert outcome['refunds'][0]['timing'] == 'immediate'
    assert outcome['refunds'][0]['amount'] == 2674.4
    assert outcome['status'] == 'cancelled'


def test_confirmed_return_receipt_does_not_claim_a_settled_refund(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
        AIMessage(content='Return requested.'),
    ])
    proposal = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156 to my original payment method.'}, config))
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation',
        'proposal_id': proposal['request']['proposal_id'], 'accepted': True}, config))
    outcome = next(json.loads(m.content.removeprefix('Application outcome: ')) for m in model.inputs[-1]
                   if m.additional_kwargs.get('application_result'))
    assert result['status'] == 'answered'
    assert outcome['return_details']['refund_status'] == 'not_settled'
    assert outcome['return_details']['additional_return_allowed'] is False
    assert outcome['return_details']['item_ids'] == ['4202497723']
    assert outcome['status'] == 'return requested'
    assert outcome['return_details']['items'] == [{'item_id': '4202497723', 'name': 'Headphones'}]


def test_claimed_return_request_without_tool_result_requires_native_proposal(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='A return request has been created for the headphones.'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156 to my original payment method.'}, config))
    assert result['status'] == 'needs_confirmation'
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'
