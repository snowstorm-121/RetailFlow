import json
from copy import deepcopy

import pytest
from retailflow.environment import RetailEnvironment
from conftest import STATE_REF

TOOL='exchange_delivered_order_items'
ARGS={'order_id':'#W2378156','item_ids':['1151293680','4983901480'],
      'new_item_ids':['7706410293','7747408585'],'payment_method_id':'credit_card_9513926'}


def authenticated(tmp_path):
    env=RetailEnvironment(tmp_path/'exchange.sqlite');env.create_session('x',STATE_REF)
    assert env.dispatch_tool('x','find_user_id_by_name_zip',{'first_name':'Yusuf','last_name':'Rossi','zip':'19122'})['ok']
    return env


def test_exchange_confirmation_display_commit_and_idempotency(tmp_path):
    env=authenticated(tmp_path);before=deepcopy(env.load_session('x')['business'])
    result=env.dispatch_tool('x',TOOL,ARGS)
    assert result['code']=='CONFIRMATION_REQUIRED'
    proposal=result['proposal'];display=proposal['display']
    assert len(display['exchange_pairs'])==2
    assert [p['new_item_id'] for p in display['exchange_pairs']]==ARGS['new_item_ids']
    assert display['payment_method_id']==ARGS['payment_method_id']
    expected=round(sum(p['new_price']-p['old_price'] for p in display['exchange_pairs']),2)
    assert display['price_difference']==expected
    assert env.load_session('x')['business']==before
    assert env.confirm_action('x',proposal['proposal_id'],True)['ok']
    committed=env.dispatch_tool('x',TOOL,ARGS)
    assert committed['ok'] and committed['data']['status']=='exchange requested'
    assert committed['data']['exchange_price_difference']==expected
    after=env.load_session('x')['business']
    assert before['users']==after['users'] and before['products']==after['products']
    assert before['orders']['#W2378156']['payment_history']==after['orders']['#W2378156']['payment_history']
    assert env.dispatch_tool('x',TOOL,ARGS)==committed
    assert env.load_session('x')['business']==after


@pytest.mark.parametrize('changes',[
    {'new_item_ids':['7706410293']},
    {'new_item_ids':['7747408585','7706410293']},
    {'payment_method_id':'someone-elses-card'},
    {'item_ids':['1151293680','1151293680'],'new_item_ids':['7706410293','7706410293']},
    {'order_id':'#W6247578'},
])
def test_invalid_exchange_cannot_create_proposal_or_write(tmp_path,changes):
    env=authenticated(tmp_path);before=env.state_hash('x')
    result=env.dispatch_tool('x',TOOL,{**ARGS,**changes})
    assert result['code'] in {'NOT_FOUND','POLICY_DENIED'}
    assert env.pending_proposal('x') is None and env.state_hash('x')==before


def test_changed_exchange_requires_new_confirmation(tmp_path):
    env=authenticated(tmp_path)
    first=env.dispatch_tool('x',TOOL,ARGS)
    assert first['code']=='CONFIRMATION_REQUIRED'
    env.confirm_action('x',first['proposal_id'],True)
    changed={**ARGS,'item_ids':['4983901480'],'new_item_ids':['7747408585']}
    second=env.dispatch_tool('x',TOOL,changed)
    assert second['code']=='CONFIRMATION_REQUIRED' and second['proposal_id']!=first['proposal_id']
    assert env.get_order('x',ARGS['order_id'])['status']=='delivered'
    assert not env.confirm_action('x',first['proposal_id'],True)['ok']


def test_exchange_dataset_oracle_checks_fields_and_side_effects(tmp_path):
    from retailflow.dataset import _make_task, check_final_state
    from retailflow.upstream.paths import RETAIL_DB_PATH
    source=json.loads((RETAIL_DB_PATH.parent/'tasks.json').read_text())
    task=next(t for t in source if t['id']=='0')
    before=json.loads(RETAIL_DB_PATH.read_text())
    derived=_make_task(task,{'user_id':'yusuf_rossi_9620','category':'legal_write','opening':'Exchange keyboard and thermostat.'},'train',before,STATE_REF.split('@')[1])
    assert derived['expected_checks']['orders'][0]['status']=='exchange requested'
    env=authenticated(tmp_path)
    p=env.dispatch_tool('x',TOOL,ARGS);env.confirm_action('x',p['proposal_id'],True);env.dispatch_tool('x',TOOL,ARGS)
    after=env.load_session('x')['business']
    events=[{'type':'confirmed_write','order_id':ARGS['order_id']},
            {'type':'assistant_message','text':' '.join(map(str,derived['expected_checks']['communicate_info']))}]
    assert check_final_state(derived,before,after,events)['passed']
    wrong=deepcopy(after);wrong['orders'][ARGS['order_id']]['exchange_price_difference']+=1
    assert not check_final_state(derived,before,wrong,events)['passed']
    wrong=deepcopy(after);wrong['orders'][ARGS['order_id']]['payment_history'].append({'transaction_type':'refund','amount':1,'payment_method_id':ARGS['payment_method_id']})
    assert not check_final_state(derived,before,wrong,events)['passed']


def test_exchange_user_rejects_wrong_options_and_payment(tmp_path):
    from retailflow.user_script import ScriptedUser
    env=authenticated(tmp_path);proposal=env.dispatch_tool('x',TOOL,ARGS)['proposal']
    consent={'tool':TOOL,'include':['Mechanical Keyboard','Smart Thermostat'],
             'exchange_required_names':['Mechanical Keyboard','Smart Thermostat'],
             'exchange_options':{'Mechanical Keyboard':{'switch type':'clicky'},'Smart Thermostat':{'compatibility':'Google Assistant'}}}
    user=ScriptedUser({'script':{'proposal_rules':[]}},consent)
    assert user.decide(proposal)==(True,None)
    wrong=deepcopy(proposal);wrong['display']['exchange_pairs'][0]['new_options']['switch type']='linear'
    assert user.decide(wrong)==(False,None)
    wrong=deepcopy(proposal);wrong['args']['payment_method_id']='another-method'
    assert user.decide(wrong)==(False,None)


def test_all_new_exchange_reference_tasks_and_user_consent(tmp_path):
    from retailflow.dataset import ROOT, check_final_state, validate_dataset
    from retailflow.upstream.paths import RETAIL_DB_PATH
    from retailflow.user_script import ScriptedUser, user_view
    cfg=json.loads((ROOT/'configs/exchange-evaluation.json').read_text());root=ROOT/cfg['dataset_root']
    assert validate_dataset(str(root/'data/manifests/dataset.json'))['counts']=={'dev':21,'test':6,'total':27}
    sources={t['id']:t for t in json.loads((RETAIL_DB_PATH.parent/'tasks.json').read_text())}
    tasks=[json.loads(line) for split in ['dev','test'] for line in (root/f'data/derived/{split}.jsonl').read_text().splitlines()]
    env=RetailEnvironment(tmp_path/'reference.sqlite')
    for task in tasks:
        if task['source_task_id'] not in {'0','1','75','77','80','93','106'}:continue
        thread=task['task_id'];env.create_session(thread,STATE_REF)
        before=env.load_session(thread)['business'];uid=task['group_id'].removeprefix('source-user:')
        assert env.dispatch_tool(thread,'find_user_id_by_email',{'email':before['users'][uid]['email']})['ok']
        user=ScriptedUser(user_view(task),cfg['user_consent'][thread]);events=[]
        for action in sources[task['source_task_id']]['evaluation_criteria']['actions']:
            result=env.dispatch_tool(thread,action['name'],action['arguments'])
            if result['code']=='CONFIRMATION_REQUIRED':
                p=result['proposal'];assert user.decide(p)==(True,None)
                env.confirm_action(thread,p['proposal_id'],True)
                result=env.dispatch_tool(thread,action['name'],action['arguments'])
                events.append({'type':'confirmed_write','order_id':action['arguments']['order_id']})
            assert result['ok'],result
        events.append({'type':'assistant_message','text':' '.join(map(str,task['expected_checks']['communicate_info']))})
        after=env.load_session(thread)['business']
        assert check_final_state(task,before,after,events)['passed']
        assert not check_final_state(task,before,before,events)['passed']


def test_exchange_rejects_unavailable_variant_and_insufficient_gift_card(tmp_path):
    env=authenticated(tmp_path);before=env.state_hash('x');db=env.load_session('x')['business']
    unavailable=next(k for k,v in db['products']['1656367028']['variants'].items() if not v['available'])
    assert env.dispatch_tool('x',TOOL,{**ARGS,'new_item_ids':[unavailable,'7747408585']})['code']=='NOT_FOUND'
    assert env.state_hash('x')==before
    # Isolated test-only account fixture: no change to the fixed source database.
    db['users']['yusuf_rossi_9620']['payment_methods']['gift_card_test']={'id':'gift_card_test','source':'gift_card','balance':0}
    order=db['orders'][ARGS['order_id']]
    next(i for i in order['items'] if i['item_id']=='1151293680')['price']=1
    order['payment_history'][0]['amount']=round(sum(i['price'] for i in order['items']),2)
    with env.storage.transaction() as sql:
        sql.execute('UPDATE sessions SET business_json=? WHERE thread=?',(json.dumps(db),'x'))
    before=env.state_hash('x')
    result=env.dispatch_tool('x',TOOL,{**ARGS,'payment_method_id':'gift_card_test'})
    assert result['code']=='POLICY_DENIED' and 'Insufficient' in result['message']
    assert env.state_hash('x')==before and env.pending_proposal('x') is None


@pytest.mark.parametrize('source_id,replacements', [
    ('0', ['6342039236','7747408585']),
    ('77', ['5081446110']),
    ('106', ['8124970213']),
])
def test_exchange_consent_rejects_variants_against_user_preferences(tmp_path, source_id, replacements):
    from retailflow.dataset import ROOT
    from retailflow.upstream.paths import RETAIL_DB_PATH
    from retailflow.user_script import ScriptedUser, user_view
    cfg = json.loads((ROOT / 'configs/exchange-evaluation.json').read_text())
    tasks = [json.loads(line) for split in ['dev', 'test']
             for line in (ROOT / cfg['dataset_root'] / f'data/derived/{split}.jsonl').read_text().splitlines()]
    task = next(t for t in tasks if t['task_id'] == f'tau-retail-{source_id}')
    source = next(t for t in json.loads((RETAIL_DB_PATH.parent / 'tasks.json').read_text()) if t['id'] == source_id)
    args = deepcopy(next(a['arguments'] for a in source['evaluation_criteria']['actions'] if a['name'] == TOOL))
    args['new_item_ids'] = replacements
    env = RetailEnvironment(tmp_path / 'consent.sqlite')
    env.create_session('x', STATE_REF)
    profile = env.load_session('x')['business']['users'][next(t['user_id'] for t in cfg['selected_tasks'] if t['id'] == source_id)]
    env.dispatch_tool('x', 'find_user_id_by_email', {'email': profile['email']})
    proposal = env.dispatch_tool('x', TOOL, args)['proposal']
    assert ScriptedUser(user_view(task), cfg['user_consent'][task['task_id']]).decide(proposal) == (False, None)


def test_exchange_runtime_displays_proposal_and_unsettled_outcome(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage, SystemMessage
    from test_agent_loop import call, runtime
    worker, model, config = runtime(tmp_path, [
        call('find_user_id_by_name_zip', {'first_name': 'Yusuf', 'last_name': 'Rossi', 'zip': '19122'}, 'identity'),
        call(TOOL, ARGS, 'exchange'),
        AIMessage(content='Your exchange is complete. An email was sent and the refund was paid.'),
    ])
    first = asyncio.run(worker.run_turn('x', {'kind': 'message', 'text': 'Yusuf Rossi, ZIP 19122. Exchange my keyboard and thermostat.'}, config))
    assert first['status'] == 'needs_confirmation'
    assert 'exchange_pairs' in first['text'] and 'price_difference' in first['text']
    assert 'exchange requested' != worker.environment.get_order('x', ARGS['order_id'])['status']
    second = asyncio.run(worker.run_turn('x', {'kind': 'confirmation',
        'proposal_id': first['request']['proposal_id'], 'accepted': True}, config))
    assert second['status'] == 'answered'
    summaries = [m.content for m in model.inputs[-1] if isinstance(m, SystemMessage)
                 and m.content.startswith('Application outcome: ')]
    summary = json.loads(summaries[-1].removeprefix('Application outcome: '))
    assert summary['status'] == 'exchange requested'
    assert summary['exchange_details']['settlement_status'] == 'not_settled'
    assert summary['exchange_details']['exchange_new_items'] == sorted(ARGS['new_item_ids'])

    assert 'This local application sends no real email' in second['text']
    assert 'not settled' in second['text']
    assert 'exchange requested' in second['text']
    assert 'An email was sent' not in second['text']
    trace = json.loads(__import__('pathlib').Path(second['trace_path']).read_text())
    assert 'An email was sent' in trace['usage']['original_model_answer']


def test_single_exchange_clarifies_which_delivered_item_before_proposing(tmp_path):
    import asyncio
    from langchain_core.messages import ToolMessage
    from test_agent_loop import call, runtime

    correct = {"order_id": "#W4073673", "item_ids": ["2216662955"], "new_item_ids": ["9844888101"],
               "payment_method_id": "credit_card_3677959"}
    wrong = {**correct, "order_id": "#W2905754", "item_ids": ["3478699712"]}
    worker, _, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity"], "question": "What is your name and ZIP?"}, "question"),
        call("find_user_id_by_name_zip", {"first_name": "Lei", "last_name": "Wilson", "zip": "32255"}, "identity"),
        call("ask_user", {"fields": ["payment_method"], "question": "Which payment method?"}, "payment"),
        call(TOOL, wrong, "wrong-source"),
        call(TOOL, correct, "proposal"),
    ])

    async def run():
        await worker.run_turn("laptop", {"kind": "message", "text":
            "I received a laptop and want to exchange it for one with an i7 processor, 8GB RAM and 1TB SSD."}, config)
        question = await worker.run_turn("laptop", {"kind": "message",
            "text": "You name is Lei Wilson and your zip code is 32255."}, config)
        assert question["status"] == "needs_input"
        assert question["request"]["fields"] == ["exchange_items"]
        assert "16GB" in question["text"] and "32GB" in question["text"]
        assert worker.environment.pending_proposal("laptop") is None

        payment = await worker.run_turn("laptop", {"kind": "message", "text": "The 15-inch laptop with 32 GB RAM."}, config)
        assert payment["request"]["fields"] == ["payment_method"]
        result = await worker.run_turn("laptop", {"kind": "message", "text": "Use my Visa."}, config)
        assert result["status"] == "needs_confirmation"
        assert result["request"]["args"] == correct
        rejected = next(m for m in worker.load_messages("laptop") if isinstance(m, ToolMessage) and m.tool_call_id == "wrong-source")
        assert json.loads(rejected.content)["code"] == "INVALID_ARGUMENT"
        assert worker.environment.get_order("laptop", correct["order_id"])["status"] == "delivered"

    asyncio.run(run())


def test_single_exchange_scope_keeps_ambiguous_answers_and_rejects_unknown_size():
    from retailflow.preproposal import single_item_exchange
    from retailflow.upstream.paths import RETAIL_DB_PATH

    business = json.loads(RETAIL_DB_PATH.read_text())
    request = "I received a laptop and want to exchange it for one with an i7 processor."
    user = "lei_wilson_4541"
    assert len(single_item_exchange(request, business, user, "The silver one.")["items"]) == 2
    assert len(single_item_exchange(request, business, user, "The 15-inch laptop with 32GB SSD.")["items"]) == 2
    assert single_item_exchange(request, business, user, "Actually, leave the laptop alone.") is None
    shirt = next(item for item in business["orders"]["#W3388163"]["items"] if item["name"] == "T-Shirt")
    shirt["options"]["size"] = "42"
    result = single_item_exchange("Exchange my T-shirt for one size smaller.", business, "sofia_thomas_1518")
    assert result["requested_size"] is None
    assert result["available_replacements"] == []


def test_relative_exchange_size_rejects_same_size_before_confirmation(tmp_path):
    import asyncio
    from langchain_core.messages import ToolMessage
    from test_agent_loop import call, runtime

    email = "sofia.thomas3069@example.com"
    correct = {"order_id": "#W3388163", "item_ids": ["9354168549"], "new_item_ids": ["2060066974"],
               "payment_method_id": "paypal_5334408"}
    worker, model, config = runtime(tmp_path, [
        call("ask_user", {"fields": ["identity"], "question": "What is your email?"}, "question"),
        call("find_user_id_by_email", {"email": email}, "identity"),
        call(TOOL, {**correct, "new_item_ids": ["3799046073"]}, "same-size"),
        call(TOOL, correct, "proposal"),
    ])

    async def run():
        await worker.run_turn("shirt", {"kind": "message", "text":
            "The T-shirt I received is too large. Exchange it for one size smaller. Keep it cotton, and I prefer black if available."}, config)
        result = await worker.run_turn("shirt", {"kind": "message", "text":
            f"You name is Sofia Thomas and your emails are sofia.thomas3019@example.com and {email}."}, config)
        assert result["status"] == "needs_confirmation"
        assert result["request"]["args"] == correct
        assert "2060066974" in model.inputs[-1][0].content
        rejected = next(m for m in worker.load_messages("shirt") if isinstance(m, ToolMessage) and m.tool_call_id == "same-size")
        assert json.loads(rejected.content)["code"] == "INVALID_ARGUMENT"
        assert worker.environment.get_order("shirt", correct["order_id"])["status"] == "delivered"

    asyncio.run(run())
