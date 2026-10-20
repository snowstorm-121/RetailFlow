"""Run real business guards directly; no model, scripted Agent, or benchmark score."""

import hashlib
import json
import uuid
from pathlib import Path

from retailflow.dataset import check_final_state
from retailflow.environment import RetailEnvironment
from retailflow.upstream.paths import RETAIL_DB_PATH


output = Path('artifacts/t8-guard') / uuid.uuid4().hex[:10]
output.mkdir(parents=True)
env = RetailEnvironment(output / 'business.sqlite')
source = RETAIL_DB_PATH.read_bytes()
thread = 'change-cancel-reason'
env.create_session(thread, 'retail-db@' + hashlib.sha256(source).hexdigest())
steps = []

# Identity and order are obtained through the same production dispatcher as the Agent.
identity = env.dispatch_tool(thread, 'find_user_id_by_name_zip',
                             {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'})
assert identity['ok']
assert env.dispatch_tool(thread, 'get_order_details', {'order_id': '#W5056519'})['ok']
before = env.load_session(thread)['business']
initial_hash = env.state_hash(thread)
original = {'order_id': '#W5056519', 'reason': 'ordered by mistake'}
first = env.dispatch_tool(thread, 'cancel_pending_order', original)
assert first['code'] == 'CONFIRMATION_REQUIRED'
assert env.confirm_action(thread, first['proposal_id'], True)['ok']
assert env.state_hash(thread) == initial_hash
steps.append({'event': 'original_approved_not_executed', 'args': original, 'proposal_id': first['proposal_id']})

# The only input change: reason. An old yes must not authorize these new parameters.
revised = {**original, 'reason': 'no longer needed'}
second = env.dispatch_tool(thread, 'cancel_pending_order', revised)
stale = env.confirm_action(thread, first['proposal_id'], True)
assert second['code'] == 'CONFIRMATION_REQUIRED'
assert second['proposal_id'] != first['proposal_id']
assert stale['code'] == 'INVALID_CONFIRMATION'
assert env.state_hash(thread) == initial_hash
steps.append({'event': 'reason_changed', 'args': revised, 'new_proposal_id': second['proposal_id'],
              'old_confirmation_result': stale, 'business_unchanged': True})

assert env.confirm_action(thread, second['proposal_id'], True)['ok']
committed = env.dispatch_tool(thread, 'cancel_pending_order', revised)
assert committed['ok']
after = env.load_session(thread)['business']
checked = check_final_state({'group_id': 'source-user:yara_muller_8652', 'expected_checks': {
    'orders': [{'order_id': revised['order_id'], 'status': 'cancelled', 'cancel_reason': revised['reason']}],
    'communicate_info': [], 'transfer_to_human': False}}, before, after,
    [{'type': 'confirmed_write', 'order_id': revised['order_id']}])
assert checked['passed'], checked
committed_hash = env.state_hash(thread)
reopened = RetailEnvironment(output / 'business.sqlite')
assert reopened.dispatch_tool(thread, 'cancel_pending_order', revised) == committed
assert reopened.state_hash(thread) == committed_hash
assert len(reopened.get_order(thread, revised['order_id'])['payment_history']) == 2
assert hashlib.sha256(RETAIL_DB_PATH.read_bytes()).digest() == hashlib.sha256(source).digest()
steps.append({'event': 'new_confirmation_committed_and_replayed', 'proposal_id': second['proposal_id'],
              'status': after['orders'][revised['order_id']]['status'],
              'payment_history_count': len(after['orders'][revised['order_id']]['payment_history']),
              'final_state_check': checked, 'replay_did_not_write_again': True})
report = {'kind': 'direct_application_guard_demo_no_model', 'steps': steps,
          'state_hash_before': initial_hash, 'state_hash_after': env.state_hash(thread),
          'source_db_unchanged': True, 'not_a_benchmark_task': True}
(output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(output), **report}, ensure_ascii=False, indent=2))
