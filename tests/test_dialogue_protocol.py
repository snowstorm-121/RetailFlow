import pytest
import asyncio
import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict

from test_agent_loop import call, runtime


FIELDS = ['identity', 'order', 'cancel_reason', 'return_items', 'refund_method', 'exchange_items', 'exchange_options', 'payment_method', 'address']


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
    assert len(model.bound_tools) == 14


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


@pytest.mark.parametrize("isolated", [False, True])
def test_information_retry_cannot_finish_with_unrelated_plain_text(tmp_path, isolated):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content='Please provide your email.'), AIMessage(content='Thanks, I can help.'),
    ])
    config['isolated_information_repair'] = isolated
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my purchase.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'information_request_not_structured'


@pytest.mark.parametrize("isolated", [False, True])
def test_information_retry_rechecks_the_quoted_request_budget(tmp_path, isolated):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        AIMessage(content='Please provide your payment method. ' + 'Details. ' * 1000),
        call('ask_user', {'fields': ['refund_method'], 'question': 'Your refund method?'}, 'should-not-run'),
    ])
    config['context_budget_tokens'] = 500
    config['isolated_information_repair'] = isolated
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
    assert len(model.bound_tools) == 14
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


@pytest.mark.parametrize("claim", [
    "A return request has been created for the headphones.",
    "A return request for the headphones has been created.",
    "A return request for the headphones was submitted.",
])
def test_claimed_return_request_without_tool_result_requires_native_proposal(tmp_path, claim):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content=claim),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156 to my original payment method.'}, config))
    assert result['status'] == 'needs_confirmation'
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'


@pytest.mark.parametrize("text", [
    "A return request for the headphones has not been created.",
    "A return request for the headphones can be created after confirmation.",
])
def test_uncreated_request_explanation_does_not_trigger_proposal_repair(tmp_path, text):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content=text),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Check #W2378156.'}, config))
    assert result['status'] == 'answered'
    assert result['usage']['dialogue_repairs'] == 0
    assert worker.environment.pending_proposal('a') is None


@pytest.mark.parametrize('text', [
    '{"action":"cancel_pending_order","order_id":"#W5056519","reason":"ordered by mistake"}',
    '```json\n{"action":"cancel_pending_order","order_id":"#W5056519"}\n```',
])
def test_plain_action_json_is_failure_not_a_completed_turn(tmp_path, text):
    worker, _, config = runtime(tmp_path, [AIMessage(content=text)])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Cancel my order.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'non_native_tool_action'
    assert result['usage']['tool_calls'] == 0
    assert worker.environment.pending_proposal('a') is None


def test_ordinary_json_answer_is_not_a_tool_action(tmp_path):
    worker, _, config = runtime(tmp_path, [AIMessage(content='{"status":"pending"}')])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Hello'}, config))
    assert result['status'] == 'answered'


@pytest.mark.parametrize("review_limit", [1, 6])
@pytest.mark.parametrize("unexpected_tool", [False, True])
def test_receipt_answer_review_is_metered_and_preserves_customer_questions(tmp_path, review_limit, unexpected_tool):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
        AIMessage(content='Refund arrives in 5 days. Remaining items can be returned later.'),
        (call('get_order_details', {'order_id': '#W2378156'}, 'unexpected') if unexpected_tool else
         AIMessage(content='Return requested; no refund date is established. No further returns for this order.')),
    ], receipt_answer_review=True)
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return headphones in #W2378156 to original payment. Can I return other items later?'}, config))
    config['max_model_calls_per_turn'] = review_limit
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation',
        'proposal_id': first['request']['proposal_id'], 'accepted': True}, config))
    if review_limit == 1:
        assert result['status'] == 'budget_exceeded'
        assert result['usage']['model_calls'] == 1
        assert worker.environment.get_order('a', '#W2378156')['status'] == 'return requested'
        return
    if unexpected_tool:
        assert result['status'] == 'failed'
        assert result['failure_reason'] == 'receipt_review_invalid_response'
        assert result['usage']['model_calls'] == 2
        assert result['usage']['tool_calls'] == 1  # Only the already confirmed write.
        return
    assert 'no refund date' in result['text']
    assert result['usage']['model_calls'] == 2
    assert result['usage']['receipt_reviews'] == 1
    assert model.bound_tools == []
    assert 'Can I return other items later?' in model.inputs[-1][-1].content
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'return requested'


def test_return_receipt_response_uses_committed_items_and_policy(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
        AIMessage(content='Refund arrives in 5 days. Remaining items can be returned later.'),
    ], return_receipt_response=True)
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return headphones in #W2378156 to original payment.'}, config))
    assert first['status'] == 'needs_confirmation'
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation',
        'proposal_id': first['request']['proposal_id'], 'accepted': True}, config))
    assert '5 days' not in result['text']
    assert 'Headphones' in result['text']
    assert 'No refund arrival date' in result['text']
    assert 'not eligible for another return' in result['text']
    assert result['usage']['model_calls'] == 1
    assert '5 days' in result['usage']['original_model_answer']
    assert worker.load_messages('a')[-1].content == result['text']


def test_isolated_information_repair_before_identity_has_no_examples(tmp_path):
    worker, model, config = runtime(tmp_path, [
        AIMessage(content='Please provide your email or full name and ZIP.'),
        call('ask_user', {'fields': ['identity'], 'question': 'Your email or full name and ZIP?'}, 'native'),
    ])
    config['isolated_information_repair'] = True
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Return my purchase.'}, config))
    retry = model.inputs[1]
    assert result['status'] == 'needs_input'
    assert len(retry) == 2
    assert retry[1].additional_kwargs.get('protocol_retry_input')
    assert not any(m.additional_kwargs.get('protocol_example') for m in retry)
    assert result['usage']['model_calls'] == 2
    assert worker.environment.pending_proposal('a') is None


def test_repeated_successfully_created_claim_without_proposal_fails(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='A return request has been created.'),
        AIMessage(content='A return request has been successfully created for your order.'),
    ])
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'proposal_not_created'
    assert result['usage']['dialogue_repairs'] == 1
    assert worker.environment.pending_proposal('a') is None
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'


def test_isolated_proposal_repair_preserves_ordered_facts_and_native_confirmation(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='A return request has been created.'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    config['isolated_proposal_repair'] = True
    text = 'Yusuf Rossi, 19122. Return only the headphones in #W2378156 to my original payment method.'
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': text}, config))
    retry = model.inputs[-1]
    assert len(retry) == 2
    events = json.loads(retry[1].content)['events']
    assert events[0] == {'role': 'customer', 'content': text}
    assert [e['role'] for e in events] == ['customer', 'assistant', 'tool', 'assistant', 'tool']
    assert events[-1]['result']['data']['order_id'] == '#W2378156'
    assert 'A return request has been created.' not in retry[1].content
    assert any(m.content == 'A return request has been created.' for m in worker.load_messages('a'))
    assert result['status'] == 'needs_confirmation'
    assert worker.environment.get_order('a', '#W2378156')['status'] == 'delivered'


def test_isolated_proposal_repair_plain_text_cannot_finish_task(tmp_path):
    worker, _, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='A return request has been created.'),
        AIMessage(content='All done.'),
    ])
    config['isolated_proposal_repair'] = True
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156.'}, config))
    assert result['status'] == 'failed'
    assert result['failure_reason'] == 'proposal_not_created'
    assert result['usage']['dialogue_repairs'] == 1
    assert worker.environment.pending_proposal('a') is None


def test_isolated_proposal_repair_keeps_later_customer_correction_last(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        call('ask_user', {'fields': ['refund_method'], 'question': 'Which refund method?'}, 'ask'),
        AIMessage(content='A return request has been created.'),
        call('return_delivered_order_items', {'order_id': '#W2378156', 'item_ids': ['4202497723'],
                                             'payment_method_id': 'credit_card_9513926'}, 'proposal'),
    ])
    config['isolated_proposal_repair'] = True
    opening = 'Yusuf Rossi, 19122. Return everything in #W2378156.'
    correction = 'Actually, only return the headphones. Use the original payment method.'
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': opening}, config))
    assert first['status'] == 'needs_input'
    second = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': correction}, config))
    events = json.loads(model.inputs[-1][1].content)['events']
    assert [e['content'] for e in events if e['role'] == 'customer'] == [opening, correction]
    assert events[-1] == {'role': 'customer', 'content': correction}
    assert events[-2]['name'] == 'ask_user'
    assert second['status'] == 'needs_confirmation'


def test_isolated_proposal_repair_does_not_extend_model_budget(tmp_path):
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call('get_order_details', {'order_id': '#W2378156'}, 'order'),
        AIMessage(content='A return request has been created.'),
    ], max_model_calls_per_turn=3)
    config['isolated_proposal_repair'] = True
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yusuf Rossi, 19122. Return the headphones in #W2378156.'}, config))
    assert result['status'] == 'budget_exceeded'
    assert len(model.inputs) == 3
    assert worker.environment.pending_proposal('a') is None


def test_staged_claim_without_order_read_requires_native_proposal(tmp_path):
    worker,_,config=runtime(tmp_path,[
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'identity'),
        AIMessage(content='The cancellation proposal is now staged and ready for confirmation.'),
        call('get_order_details',{'order_id':'#W5056519'},'order'),
        call('cancel_pending_order',{'order_id':'#W5056519','reason':'ordered by mistake'},'proposal'),
    ],isolated_proposal_repair=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':'Yara Muller, 85041. Cancel #W5056519, ordered by mistake.'},config))
    assert result['status']=='needs_confirmation'
    assert result['usage']['dialogue_repairs']==1
    assert worker.environment.get_order('a','#W5056519')['status']=='pending'


@pytest.mark.parametrize('text',[
    'The cancellation proposal is not staged.',
    'The cancellation proposal will be staged after the order is checked.',
    'Once the cancellation proposal is staged, you will be asked to confirm it.',
])
def test_unstaged_explanation_without_order_read_is_not_a_claim(tmp_path,text):
    worker,_,config=runtime(tmp_path,[
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'identity'),
        AIMessage(content=text),
    ])
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':'Yara Muller, 85041. Check my account.'},config))
    assert result['status']=='answered'
    assert result['usage']['dialogue_repairs']==0


@pytest.mark.parametrize('native_finish', [False, True])
def test_proposal_recovery_survives_intermediate_read(tmp_path, native_finish):
    final = (call('cancel_pending_order', {'order_id': '#W5056519', 'reason': 'ordered by mistake'}, 'proposal')
             if native_finish else AIMessage(content='The cancellation is being processed.'))
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        AIMessage(content='The cancellation proposal is now staged.'),
        call('get_order_details', {'order_id': '#W5056519'}, 'read'),
        final,
    ], isolated_proposal_repair=True)
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text':
        'Yara Muller, 85041. Cancel #W5056519, ordered by mistake.'}, config))
    assert model.inputs[-1][-1].additional_kwargs.get('protocol_retry_input')
    assert 'The cancellation proposal is now staged.' not in model.inputs[-1][-1].content
    assert 'get_order_details' in model.inputs[-1][-1].content
    assert result['status'] == ('needs_confirmation' if native_finish else 'failed')
    if not native_finish:
        assert result['failure_reason'] == 'proposal_not_created'
    assert result['usage']['dialogue_repairs'] == 1
    assert result['usage']['model_calls'] == 4
    assert worker.environment.get_order('a', '#W5056519')['status'] == 'pending'


@pytest.mark.parametrize("draft", [
    "1. **Confirm which 13-inch laptop variant you prefer**.\n"
    "2. **Confirm if you want to proceed with returning the non-water-resistant speaker**.\n"
    "Once you confirm both, I will generate the proposals.",
    "Please confirm:\n- ✅ Return the non-water-resistant speaker\n"
    "- 🔍 Choose which item variant you prefer for the laptop.",
])
def test_mixed_confirmation_draft_recovers_ready_return_before_laptop_choice(tmp_path, draft):
    email = "aarav.santos8320@example.com"
    speaker_order, laptop_order = "#W8528674", "#W9672333"
    args = {"order_id": speaker_order, "item_ids": ["6704763132"], "payment_method_id": "paypal_7664977"}
    worker, model, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_order_details", {"order_id": laptop_order}, "laptop"),
        call("get_order_details", {"order_id": speaker_order}, "speaker"),
        AIMessage(content=draft),
        call("return_delivered_order_items", args, "return"),
        call("ask_user", {"fields": ["exchange_options"], "question": "Which eligible laptop variant do you prefer?"}, "variant"),
    ], isolated_proposal_repair=True)

    async def run():
        first = await worker.run_turn("mixed-return", {"kind": "message", "text":
            f"My email is {email}. Return the more expensive speaker that is not water resistant. "
            "Modify the 17-inch laptop in another order to 13-inch."}, config)
        assert first["status"] == "needs_confirmation"
        assert first["request"]["name"] == "return_delivered_order_items"
        assert first["request"]["args"] == args
        assert first["usage"]["dialogue_repairs"] == 1
        assert draft not in model.inputs[-1][-1].content
        assert "Handle the independently requested return first" in model.inputs[-1][0].content
        assert worker.environment.get_order("mixed-return", speaker_order)["status"] == "delivered"
        assert worker.environment.receipt_for_proposal("mixed-return", first["request"]["proposal_id"]) is None

        followup = await worker.run_turn("mixed-return", {"kind": "confirmation", "accepted": True,
                                                         "proposal_id": first["request"]["proposal_id"]}, config)
        assert followup["status"] == "needs_input"
        assert followup["request"]["fields"] == ["exchange_options"]
        assert worker.environment.get_order("mixed-return", speaker_order)["status"] == "return requested"
        assert worker.environment.get_order("mixed-return", laptop_order)["status"] == "pending"

    asyncio.run(run())



@pytest.mark.parametrize("request_text", [
    "Is the speaker in #W8528674 eligible for a return?",
    "Return the speaker in #W9672333. Check order #W8528674 too.",
    "Return the speaker only if you can modify my laptop to 13 inches.",
    "Return the speaker from #W9672333, not #W8528674.",
    "Return the speaker. Only do that if you can modify my laptop.",
    "Return the speaker and exchange the headphones. If only one option is possible, choose whichever saves more.",
    "Return the speaker and exchange the headphones. Choose whichever is cheaper.",
    "Return the speaker and exchange the headphones. Pick the one that saves the most.",
    "Return the laptop. Check the speaker in #W8528674 too.",
])
def test_return_confirmation_repair_requires_independent_eligible_scope(tmp_path, request_text):
    email, order_id = "aarav.santos8320@example.com", "#W8528674"
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_order_details", {"order_id": order_id}, "speaker"),
        AIMessage(content="Confirm if you want to proceed with returning the speaker."),
    ], isolated_proposal_repair=True)
    result = asyncio.run(worker.run_turn("return-question", {"kind": "message", "text":
        f"My email is {email}. {request_text}"}, config))
    assert result["status"] == "answered"
    assert result["usage"]["dialogue_repairs"] == 0
    assert worker.environment.pending_proposal("return-question") is None


@pytest.mark.parametrize(("recover_commit", "extra_request", "stale_catalog"), [
    (False, "", False), (True, "", False), (False, " Explain why my previous refund was declined.", False),
    (False, "", True),
])
def test_mixed_write_reply_uses_actual_replacement_and_both_receipts(tmp_path, recover_commit, extra_request, stale_catalog):
    email = "aarav.santos8320@example.com"
    speaker_order, laptop_order = "#W8528674", "#W9672333"
    returned = {"order_id": speaker_order, "item_ids": ["6704763132"], "payment_method_id": "paypal_7664977"}
    modified = {"order_id": laptop_order, "item_ids": ["1684786391"], "new_item_ids": ["5052031638"],
                "payment_method_id": "paypal_7664977"}
    draft = "The new laptop has 32GB RAM. The difference is $101.31, charged to your PayPal account. Both are still proposals."
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_order_details", {"order_id": speaker_order}, "speaker"),
        call("return_delivered_order_items", returned, "return"),
        call("get_order_details", {"order_id": laptop_order}, "laptop"),
        call("get_product_details", {"product_id": "4760268021"}, "variants"),
        call("modify_pending_order_items", modified, "modify"),
        AIMessage(content=draft),
        AIMessage(content="How can I help with your new question?"),
    ], return_receipt_response=True)

    async def run():
        thread = "mixed-receipts"
        first = await worker.run_turn(thread, {"kind": "message", "text":
            f"My email is {email}. Return the more expensive speaker that is not water resistant. "
            "Modify the 17-inch laptop in another order to 13-inch. "
            "If the exact version is unavailable, list all available 13-inch options." + extra_request}, config)
        assert first["status"] == "needs_confirmation"
        second = await worker.run_turn(thread, {"kind": "confirmation", "accepted": True,
                                               "proposal_id": first["request"]["proposal_id"]}, config)
        assert second["status"] == "needs_confirmation"
        proposal_id = second["request"]["proposal_id"]
        if stale_catalog:
            business = worker.environment.load_session(thread)["business"]
            business["products"]["4760268021"]["variants"]["1657832319"]["available"] = False
            with worker.environment.storage.transaction() as db:
                db.execute("UPDATE sessions SET business_json = ? WHERE thread = ?", (json.dumps(business), thread))
        if recover_commit:
            assert worker.environment.confirm_action(thread, proposal_id, True)["ok"]
            assert worker.environment.dispatch_tool(thread, "modify_pending_order_items", modified)["ok"]
        final = await worker.run_turn(thread, {"kind": "confirmation", "accepted": True, "proposal_id": proposal_id},
                                      config)
        assert final["status"] == "answered"
        if extra_request:
            assert final["text"] == draft
            return
        assert final["usage"]["original_model_answer"] == draft
        for fact in ("Bluetooth Speaker", "return requested", "PayPal", "13-inch", "16GB", "113.71",
                     "No real payment was sent", "5052031638"):
            assert fact.lower() in final["text"].lower()
        if stale_catalog:
            assert "alternatives could not be verified" in final["text"]
            assert "1657832319" not in final["text"]
        else:
            assert "6056040996" in final["text"] and "1657832319" in final["text"]
        assert "101.31" not in final["text"]
        assert "still proposals" not in final["text"]
        assert "charged to your PayPal" not in final["text"]

        next_request = await worker.run_turn(thread, {"kind": "message", "text": "I have a new question."}, config)
        assert next_request["text"] == "How can I help with your new question?"

    asyncio.run(run())


@pytest.mark.parametrize("condition", ["", " Only do that if there is no price difference."])
def test_confirmed_return_continues_to_independent_pending_item_change(tmp_path, condition):
    email = "aarav.santos8320@example.com"
    speaker_order, laptop_order = "#W8528674", "#W9672333"
    returned = {"order_id": speaker_order, "item_ids": ["6704763132"], "payment_method_id": "paypal_7664977"}
    modified = {"order_id": laptop_order, "item_ids": ["1684786391"], "new_item_ids": ["5052031638"],
                "payment_method_id": "paypal_7664977"}
    worker, _, config = runtime(tmp_path, [
        call("find_user_id_by_email", {"email": email}, "identity"),
        call("get_order_details", {"order_id": speaker_order}, "speaker"),
        call("return_delivered_order_items", returned, "return"),
        AIMessage(content="The speaker return is complete."),
        call("get_order_details", {"order_id": laptop_order}, "laptop"),
        call("modify_pending_order_items", modified, "premature"),
        call("get_product_details", {"product_id": "4760268021"}, "catalog"),
        call("modify_pending_order_items", {**modified, "item_ids": ["3714494375"]}, "wrong-item"),
        call("modify_pending_order_items", modified, "modify"),
        AIMessage(content="Both confirmed actions are recorded."),
    ], return_receipt_response=True, max_model_calls_per_turn=8)

    async def run():
        thread = "mixed-continuation"
        first = await worker.run_turn(thread, {"kind": "message", "text":
            f"My email is {email}. Return the more expensive speaker that is not water resistant. "
            "Modify the 17-inch laptop in another order to 13-inch." + condition}, config)
        assert first["status"] == "needs_confirmation"
        second = await worker.run_turn(thread, {"kind": "confirmation", "accepted": True,
                                               "proposal_id": first["request"]["proposal_id"]}, config)
        if condition:
            assert second["status"] == "answered"
            assert worker.environment.pending_proposal(thread) is None
            assert worker.environment.get_order(thread, laptop_order)["status"] == "pending"
            return
        assert second["status"] == "needs_confirmation"
        assert second["request"]["args"] == modified
        assert second["usage"]["focused_prose_repairs"] == 1
        premature = next(message for message in worker.load_messages(thread)
                         if isinstance(message, ToolMessage) and message.tool_call_id == "premature")
        assert "Read the current catalog" in json.loads(premature.content)["message"]
        wrong_item = next(message for message in worker.load_messages(thread)
                          if isinstance(message, ToolMessage) and message.tool_call_id == "wrong-item")
        assert wrong_item.additional_kwargs["focused_write_mismatch"] and wrong_item.status == "error"
        proposal_id = second["request"]["proposal_id"]
        assert first["request"]["proposal_id"] != proposal_id
        assert worker.environment.get_order(thread, speaker_order)["status"] == "return requested"
        assert worker.environment.get_order(thread, laptop_order)["status"] == "pending"
        assert worker.environment.receipt_for_proposal(thread, proposal_id) is None
        final = await worker.run_turn(thread, {"kind": "confirmation", "accepted": True,
                                              "proposal_id": proposal_id}, config)
        assert final["status"] == "answered"
        assert "Item modification confirmed" in final["text"]
        assert worker.environment.get_order(thread, laptop_order)["status"] == "pending (item modified)"

    asyncio.run(run())


@pytest.mark.parametrize("prior_goal", [False, True])
def test_cancelled_order_reinstatement_recovers_to_authenticated_local_handoff(tmp_path, prior_goal):
    worker, _, config = runtime(tmp_path, [
        AIMessage(content="I cannot reopen a cancelled order."),
        call("ask_user", {"fields": ["order"], "question": "What is the order number?"}, "irrelevant-question"),
        call("ask_user", {"fields": ["identity"], "question": "What is your name and ZIP?"}, "question"),
        call("find_user_id_by_name_zip", {"first_name": "Yara", "last_name": "Muller", "zip": "85041"}, "identity"),
        AIMessage(content="A cancelled order cannot be reopened."),
        call("transfer_to_human_agents", {"summary": "Customer wants to restore a cancelled order."}, "handoff"),
    ], cancel_goal_tracking=prior_goal, cancel_goal_activation="explicit_all_pending")

    async def run():
        thread = "reinstate"
        if prior_goal:
            worker._save_runtime(thread, {"started_at": time.time(), "model_calls": 0, "user_turns": 0,
                                          "cancel_goal": {"needs_refresh": True}})
        first = await worker.run_turn(thread, {"kind": "message", "text":
            "Please undo an order I cancelled and make sure it gets delivered as soon as possible."}, config)
        assert first["status"] == "needs_input"
        assert first["request"]["fields"] == ["identity"]
        unrelated = next(message for message in worker.load_messages(thread)
                         if isinstance(message, ToolMessage) and message.tool_call_id == "irrelevant-question")
        assert json.loads(unrelated.content)["code"] == "INVALID_ARGUMENT"
        result = await worker.run_turn(thread, {"kind": "message", "text":
            "My name is Yara Muller, ZIP 85041."}, config)
        assert result["status"] == "handed_off"
        assert "cannot reinstate a cancelled order" in result["text"].lower()
        assert "no human" in result["text"].lower()
        trace = json.loads(Path(result["trace_path"]).read_text())
        assert trace["state_hash_before"] == trace["state_hash_after"]
        assert worker.environment.pending_proposal(thread) is None

    asyncio.run(run())


def test_question_about_reinstatement_does_not_trigger_handoff(tmp_path):
    worker, _, config = runtime(tmp_path, [AIMessage(content="A cancelled order cannot be restored.")])
    result = asyncio.run(worker.run_turn("reinstatement-question", {"kind": "message",
        "text": "Can you tell me whether a cancelled order can be reinstated?"}, config))
    assert result["status"] == "answered"
    assert result["usage"]["model_calls"] == 1
