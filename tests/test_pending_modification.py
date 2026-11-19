import json
from copy import deepcopy
import pytest
from conftest import STATE_REF
from retailflow.environment import RetailEnvironment
from retailflow.upstream.paths import RETAIL_DB_PATH
from retailflow.upstream.data_model import RetailDB
from retailflow.upstream.tools import RetailTools

SOURCE = json.loads((RETAIL_DB_PATH.parent / 'tasks.json').read_text())
DB = json.loads(RETAIL_DB_PATH.read_text())
TOOL = 'modify_pending_order_items'


def modification(source_id='20'):
    task = next(t for t in SOURCE if t['id'] == source_id)
    return next(a['arguments'] for a in task['evaluation_criteria']['actions'] if a['name'] == TOOL)


def test_upstream_multi_item_modification_preserves_each_variant():
    db = RetailDB.model_validate(DB)
    args = modification()
    before = deepcopy(DB['orders'][args['order_id']])
    result = RetailTools(db).modify_pending_order_items(**args).model_dump(mode='json')
    for old_id, new_id in zip(args['item_ids'], args['new_item_ids']):
        old = next(i for i in before['items'] if i['item_id'] == old_id)
        variant = DB['products'][old['product_id']]['variants'][new_id]
        actual = next(i for i in result['items'] if i['item_id'] == new_id)
        assert actual['price'] == variant['price']
        assert actual['options'] == variant['options']


def test_pending_modification_confirmation_difference_and_gift_balance(tmp_path):
    env = RetailEnvironment(tmp_path / 'db.sqlite'); env.create_session('x', STATE_REF)
    args = modification(); uid = DB['orders'][args['order_id']]['user_id']
    env.dispatch_tool('x', 'find_user_id_by_email', {'email': DB['users'][uid]['email']})
    before = env.state_hash('x')
    p = env.dispatch_tool('x', TOOL, args)
    assert p['code'] == 'CONFIRMATION_REQUIRED'
    display = p['proposal']['display']
    assert len(display['exchange_pairs']) == 4 and env.state_hash('x') == before
    env.confirm_action('x', p['proposal_id'], True)
    result = env.dispatch_tool('x', TOOL, args)
    assert result['data']['status'] == 'pending (item modified)'
    assert env.dispatch_tool('x', TOOL, args) == result
    after = env.load_session('x')['business']
    method = args['payment_method_id']
    assert after['users'][uid]['payment_methods'][method]['balance'] == round(DB['users'][uid]['payment_methods'][method]['balance'] - display['price_difference'], 2)
    other = {**args, 'new_item_ids': list(reversed(args['new_item_ids']))}
    assert not env.dispatch_tool('x', TOOL, other)['ok']


@pytest.mark.parametrize('expression', ['2**100000000', '__import__("os")', '1/0', '9'*200, '1e309', '(1+2j)', '1//2'])
def test_calculate_rejects_unsafe_or_unbounded_expressions(expression):
    from retailflow.tools import invoke
    with pytest.raises(ValueError):
        invoke(RetailDB.model_validate(DB), 'calculate', {'expression': expression})


def test_calculate_normal_amount_and_no_mutation():
    from retailflow.tools import invoke
    db = RetailDB.model_validate(DB); before = db.model_dump()
    assert invoke(db, 'calculate', {'expression': '3131.1 + 4777.75 + 367.38'}) == '8276.23'
    assert db.model_dump() == before


def test_modification_oracle_rejects_wrong_items_payments_and_balance(tmp_path):
    from retailflow.dataset import _make_task, check_final_state
    source = next(t for t in SOURCE if t['id'] == '20'); args = modification()
    uid = DB['orders'][args['order_id']]['user_id']
    task = _make_task(source, {'user_id': uid, 'category': 'legal_write', 'opening': 'Upgrade my items.'}, 'train', DB, STATE_REF.split('@')[1])
    env = RetailEnvironment(tmp_path / 'oracle.sqlite');env.create_session('x',STATE_REF)
    env.dispatch_tool('x','find_user_id_by_email',{'email':DB['users'][uid]['email']})
    p=env.dispatch_tool('x',TOOL,args);env.confirm_action('x',p['proposal_id'],True);env.dispatch_tool('x',TOOL,args)
    after=env.load_session('x')['business'];events=[{'type':'confirmed_write','order_id':args['order_id']}]
    assert check_final_state(task,DB,after,events)['passed']
    assert not check_final_state(task,DB,DB,[])['passed']
    wrong=deepcopy(after);wrong['orders'][args['order_id']]['items'][0]['price']+=1
    assert not check_final_state(task,DB,wrong,events)['passed']
    wrong=deepcopy(after);wrong['users'][uid]['payment_methods'][args['payment_method_id']]['balance']+=1
    assert not check_final_state(task,DB,wrong,events)['passed']


def test_consent_scopes_tools_and_preserves_duplicate_item_requirements():
    from retailflow.user_script import ScriptedUser
    spec = {'script': {'proposal_rules': []}}
    consent = {'by_tool': {TOOL: {'include': ['Tea Kettle'], 'exchange_requirements': [
        {'name': 'Tea Kettle', 'new_options': {'material': 'ceramic'}},
        {'name': 'Tea Kettle', 'new_options': {'capacity': '1.5 liters'}},
    ]}}}
    p = {'name': TOOL, 'args': {'order_id': '#W1', 'item_ids': ['a','a'], 'new_item_ids': ['b','c'], 'payment_method_id':'card'},
         'display': {'items':[{'name':'Tea Kettle','item_id':'a'}], 'original_payment_method_id':'card',
         'exchange_pairs': [{'name':'Tea Kettle','old_options':{},'new_options':{'material':'ceramic'}},
                            {'name':'Tea Kettle','old_options':{},'new_options':{'capacity':'1.5 liters'}}]}}
    user = ScriptedUser(spec, consent)
    assert user.decide(p) == (True, None)
    wrong=deepcopy(p);wrong['display']['exchange_pairs'][1]['new_options']={'material':'ceramic'}
    assert user.decide(wrong) == (False, None)
    wrong=deepcopy(p);wrong['name']='return_delivered_order_items'
    assert user.decide(wrong)==(False,None)


def test_modification_uses_original_rows_when_replacement_is_another_old_id():
    from retailflow.dataset import _modification_check
    args={'order_id':'#W9318778','item_ids':['5669664287','3076708684'],
          'new_item_ids':['3076708684','8302289002'],'payment_method_id':'credit_card_7862034'}
    expected=_modification_check(args,DB)
    actual=RetailTools(RetailDB.model_validate(DB)).modify_pending_order_items(**args).model_dump(mode='json')
    assert actual['items']==expected['items']


def test_resumed_cli_shows_modification_details_before_confirmation(tmp_path, monkeypatch, capsys):
    import asyncio
    from retailflow.cli import chat
    env=RetailEnvironment(tmp_path/'cli.sqlite');env.create_session('x',STATE_REF)
    args=modification();uid=DB['orders'][args['order_id']]['user_id']
    env.dispatch_tool('x','find_user_id_by_email',{'email':DB['users'][uid]['email']})
    env.dispatch_tool('x',TOOL,args)
    config=tmp_path/'config.json';config.write_text(json.dumps({'store_path':str(env.store_path),'trace_dir':str(tmp_path/'traces')}))
    monkeypatch.setattr('builtins.input',lambda _: 'later')
    assert asyncio.run(chat('x',config))==0
    output=capsys.readouterr().out
    assert 'exchange_pairs' in output and 'price_difference' in output
    assert env.pending_proposal('x')['confirmed'] is False


@pytest.mark.parametrize('reverse',[False,True])
def test_overlapping_preferences_match_independently_of_pair_order(reverse):
    from retailflow.user_script import ScriptedUser
    pairs=[{'name':'Tea Kettle','old_options':{},'new_options':{'material':'ceramic','capacity':'1.5 liters'}},
           {'name':'Tea Kettle','old_options':{},'new_options':{'material':'ceramic','capacity':'2 liters'}}]
    if reverse:pairs.reverse()
    p={'name':TOOL,'args':{'order_id':'x','item_ids':['a','a'],'new_item_ids':['b','c'],'payment_method_id':'card'},
       'display':{'items':[{'name':'Tea Kettle'}],'original_payment_method_id':'card','exchange_pairs':pairs}}
    c={'tool':TOOL,'include':['Tea Kettle'],'exchange_requirements':[{'name':'Tea Kettle','new_options':{'material':'ceramic'}},{'name':'Tea Kettle','new_options':{'capacity':'1.5 liters'}}]}
    assert ScriptedUser({'script':{'proposal_rules':[]}},c).decide(p)==(True,None)


def test_modification_text_confirmation_recovers_to_native_proposal(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import call,runtime
    args=modification('44')
    worker,model,config=runtime(tmp_path,[
        call('find_user_id_by_name_zip',{'first_name':'Aarav','last_name':'Anderson','zip':'19031'},'identity'),
        call('get_order_details',{'order_id':args['order_id']},'order'),
        AIMessage(content='I will modify this item. Please confirm if you want this change.'),
        call(TOOL,args,'modify'),
    ],isolated_proposal_repair=True)
    r=asyncio.run(worker.run_turn('x',{'kind':'message','text':'I am Aarav Anderson, ZIP 19031. Modify my desk lamp to the cheapest one and refund my gift card.'},config))
    assert r['status']=='needs_confirmation'
    assert r['usage']['dialogue_repairs']==1
    assert worker.environment.get_order('x',args['order_id'])['status']=='pending'
