import json

from retailflow.environment import RetailEnvironment
from retailflow.upstream.paths import RETAIL_DB_PATH
from conftest import STATE_REF


def test_address_changes_are_confirmed_and_scoped(tmp_path):
    before = json.loads(RETAIL_DB_PATH.read_text())
    order_id = '#W8665881'
    uid = before['orders'][order_id]['user_id']
    env = RetailEnvironment(tmp_path / 'data.sqlite')
    env.create_session('x', STATE_REF)
    env.dispatch_tool('x', 'find_user_id_by_email', {'email': before['users'][uid]['email']})
    address = {**before['orders'][order_id]['address'], 'address2': ''}
    order_args = {'order_id': order_id, **address}
    profile_args = {'user_id': uid, **address}
    assert not env.dispatch_tool('x', 'modify_user_address', {**profile_args, 'user_id': next(u for u in before['users'] if u != uid)})['ok']
    for name, args in [('modify_pending_order_address', order_args), ('modify_user_address', profile_args)]:
        snapshot = env.state_hash('x')
        proposed = env.dispatch_tool('x', name, args)
        assert proposed['code'] == 'CONFIRMATION_REQUIRED'
        assert proposed['proposal']['display']['new_address'] == address
        assert env.state_hash('x') == snapshot
        assert env.confirm_action('x', proposed['proposal_id'], True)['ok']
        committed = env.dispatch_tool('x', name, args)
        assert committed['ok']
        assert env.dispatch_tool('x', name, args) == committed
    after = env.load_session('x')['business']
    before['orders'][order_id]['address'] = address
    before['users'][uid]['address'] = address
    assert after == before


def test_all_new_address_tasks_execute_and_reject_missing_confirmation(tmp_path):
    from retailflow.dataset import ROOT, check_final_state, validate_dataset
    from retailflow.user_script import ScriptedUser, user_view

    config = json.loads((ROOT / 'configs/address-evaluation.json').read_text())
    root = ROOT / config['dataset_root']
    validated = validate_dataset(root / 'data/manifests/dataset.json')
    assert validated['valid'], validated
    assert validated['counts'] == {'dev': 53, 'test': 20, 'total': 73}
    env = RetailEnvironment(tmp_path / 'references.sqlite')
    seen = 0
    for split in ('dev', 'test'):
        for line in (root / f'data/derived/{split}.jsonl').read_text().splitlines():
            task = json.loads(line)
            thread = task['task_id']
            consent = config['user_consent'].get(thread, {})
            actions = consent.get('approved_reference_actions')
            if actions is None:
                continue
            seen += 1
            env.create_session(thread, task['initial_state_ref'])
            before = env.load_session(thread)['business']
            uid = task['group_id'].removeprefix('source-user:')
            assert env.dispatch_tool(thread, 'find_user_id_by_email', {'email': before['users'][uid]['email']})['ok']
            user = ScriptedUser(user_view(task), consent)
            events = [{'type': 'assistant_message', 'text': ' '.join(map(str, task['expected_checks']['communicate_info']))}]
            for action in actions:
                name, args = action['name'], action['arguments']
                proposal = env.dispatch_tool(thread, name, args)
                assert proposal['code'] == 'CONFIRMATION_REQUIRED', (thread, proposal)
                assert user.decide(proposal['proposal']) == (True, None)
                assert env.confirm_action(thread, proposal['proposal_id'], True)['ok']
                commit = env.dispatch_tool(thread, name, args)
                assert commit['ok'], (thread, commit)
                events.append({'type': 'confirmed_write', 'name': name, 'args': args, 'order_id': args.get('order_id')})
            after = env.load_session(thread)['business']
            checked = check_final_state(task, before, after, events)
            assert checked['passed'], (thread, checked)
            assert 'missing_confirmation' in check_final_state(task, before, after, events[1:-1])['failures']
            assert not check_final_state(task, before, before, [])['passed']
            if task['source_task_id'] == '17':
                after['orders']['#W8665881']['return_items'] = ['unexpected']
                assert 'target_extra_change' in check_final_state(task, before, after, events)['failures']
    assert seen == 18


def test_old_address_receipt_cannot_authorize_restoring_an_address(tmp_path):
    data = json.loads(RETAIL_DB_PATH.read_text())
    uid = data['orders']['#W8665881']['user_id']
    profile = data['users'][uid]
    env = RetailEnvironment(tmp_path / 'data.sqlite')
    env.create_session('x', STATE_REF)
    env.dispatch_tool('x', 'find_user_id_by_email', {'email': profile['email']})
    name = 'modify_user_address'
    for suite in ('Suite 1', 'Suite 2'):
        args = {'user_id': uid, **profile['address'], 'address2': suite}
        proposal = env.dispatch_tool('x', name, args)
        env.confirm_action('x', proposal['proposal_id'], True)
        assert env.dispatch_tool('x', name, args)['ok']
    restore = {'user_id': uid, **profile['address'], 'address2': 'Suite 1'}
    proposal = env.dispatch_tool('x', name, restore)
    assert proposal['code'] == 'CONFIRMATION_REQUIRED'
    assert env.load_session('x')['business']['users'][uid]['address']['address2'] == 'Suite 2'
    assert env.confirm_action('x', proposal['proposal_id'], True)['ok']
    restored = env.dispatch_tool('x', name, restore)
    assert restored['data']['address']['address2'] == 'Suite 1'
    assert restored['proposal_id'] == proposal['proposal_id']
    assert env.dispatch_tool('x', name, restore) == restored


def test_evaluator_collects_profile_receipt_without_order_id(tmp_path):
    import asyncio

    from langchain_core.messages import AIMessage
    from retailflow.dataset import ROOT
    from retailflow.evaluation import evaluate_episode
    from test_agent_loop import call, runtime

    config = json.loads((ROOT / 'configs/address-evaluation.json').read_text())
    root = ROOT / config['dataset_root'] / 'data/derived'
    task = next(t for path in root.glob('*.jsonl') for line in path.read_text().splitlines()
                if (t := json.loads(line))['source_task_id'] == '33')
    name = 'modify_user_address'
    args = task['expected_checks']['address_actions'][0]['arguments']
    task['initial_user_message'] += ' My name is Noah Patel, ZIP 10108.'
    worker, _, execution = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Noah', 'last_name': 'Patel', 'zip': '10108'}, 'identity'),
        call(name, args, 'proposal'),
        AIMessage(content='I updated your default address. Your order was left unchanged.'),
    ])
    result = asyncio.run(evaluate_episode(task, 'B2', {
        **config, 'execution': execution, 'output_dir': str(tmp_path), 'runtime': worker,
    }))
    assert result['objective_passed'], result
    receipt = next(event for event in result['events'] if event['type'] == 'confirmed_write')
    assert receipt['name'] == name and receipt['order_id'] is None


def test_mixed_address_return_task_preserves_tracking_answer(tmp_path):
    import asyncio

    from langchain_core.messages import AIMessage
    from retailflow.dataset import ROOT
    from retailflow.evaluation import evaluate_episode
    from test_agent_loop import call, runtime

    config = json.loads((ROOT / 'configs/address-evaluation.json').read_text())
    root = ROOT / config['dataset_root'] / 'data/derived'
    task = next(t for path in root.glob('*.jsonl') for line in path.read_text().splitlines()
                if (t := json.loads(line))['source_task_id'] == '104')
    task['initial_user_message'] += ' My email is lucas.brown9344@example.com.'
    actions = config['user_consent'][task['task_id']]['approved_reference_actions']
    worker, _, execution = runtime(tmp_path, [
        call('find_user_id_by_email', {'email': 'lucas.brown9344@example.com'}, 'identity'),
        *(call(action['name'], action['arguments'], f'proposal-{i}') for i, action in enumerate(actions)),
        AIMessage(content='The cancelled order tracking number is 286422338955.'),
    ], return_receipt_response=True)
    result = asyncio.run(evaluate_episode(task, 'B2', {
        **config, 'execution': execution, 'output_dir': str(tmp_path), 'runtime': worker,
    }))
    assert result['objective_passed'] and result['required_info_literal_match'], result
