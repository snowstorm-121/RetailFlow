import json
from copy import deepcopy
from pathlib import Path
from retailflow.dataset import ROOT, prepare_dataset, validate_dataset, check_final_state


def test_explicit_group_assignment_preserves_upstream_provenance(tmp_path):
    cfg=json.loads((ROOT/'configs/exchange-evaluation.json').read_text())
    selected=next(t for t in cfg['selected_tasks'] if t['id']=='77')
    cfg['selected_tasks']=[selected];cfg['output_root']=str(tmp_path)
    cfg['group_assignments']={selected['user_id']:'dev'}
    result=prepare_dataset(cfg)
    assert result['counts']=={'dev':1,'test':0,'total':1}
    assert validate_dataset(tmp_path/'data/manifests/dataset.json')['valid']
    task=json.loads((tmp_path/'data/derived/dev.jsonl').read_text())
    assert task['source_split']=='test' and task['split']=='dev'


def test_required_user_revision_cannot_be_skipped():
    root=ROOT/'data/versions/t3-exchange-v1'
    task=json.loads((root/'data/derived/dev.jsonl').read_text().splitlines()[0])
    task['script']['required_proposal_rules']=[0]
    task['expected_checks']={'orders':[],'communicate_info':[],'transfer_to_human':False}
    before={'users':{},'products':{},'orders':{}}
    assert 'missing_user_revision' in check_final_state(task,before,before,[])['failures']
    assert check_final_state(task,before,before,[{'type':'user_revision','rule_indices':[0]}])['passed']


def test_every_new_source_reference_has_valid_consent_state_and_negative_control(tmp_path):
    from retailflow.environment import RetailEnvironment
    from retailflow.tools import WRITE_TOOLS
    from retailflow.upstream.paths import RETAIL_DB_PATH
    from retailflow.user_script import ScriptedUser, user_view
    cfg=json.loads((ROOT/'configs/coverage-evaluation.json').read_text())
    root=ROOT/cfg['dataset_root']
    checked=validate_dataset(root/'data/manifests/dataset.json')
    assert checked['valid'], checked
    assert checked['counts']=={'dev':46,'test':9,'total':55}
    previous={s['id'] for s in json.loads((ROOT/'configs/exchange-evaluation.json').read_text())['selected_tasks']}
    sources={t['id']:t for t in json.loads((RETAIL_DB_PATH.parent/'tasks.json').read_text())}
    tasks=[json.loads(l) for split in ['dev','test'] for l in (root/f'data/derived/{split}.jsonl').read_text().splitlines()]
    env=RetailEnvironment(tmp_path/'references.sqlite')
    for task in tasks:
        id=task['source_task_id']
        if id in previous:continue
        thread=task['task_id'];env.create_session(thread,task['initial_state_ref'])
        before=env.load_session(thread)['business'];uid=task['group_id'].removeprefix('source-user:')
        assert env.dispatch_tool(thread,'find_user_id_by_email',{'email':before['users'][uid]['email']})['ok']
        user=ScriptedUser(user_view(task),cfg['user_consent'][thread]);events=[]
        # Reference traces omit rejected proposals; exercise each registered revision before final actions.
        for index,rule in enumerate(task['script']['proposal_rules']):
            # Real eligible source order; pre-final rejected proposal uses available business variants.
            order_id='#W6390527'
            args={'order_id':order_id,'item_ids':['8384507844'], 'new_item_ids':['7453605304'], 'payment_method_id':'paypal_7644869'}
            result=env.dispatch_tool(thread,rule['tool'],args)
            assert result['code']=='CONFIRMATION_REQUIRED',(id,result)
            accepted,reply=user.decide(result['proposal']);assert not accepted and reply
            assert env.confirm_action(thread,result['proposal_id'],False)['ok']
            assert env.load_session(thread)['business']==before
            events.append({'type':'user_revision','rule_indices':[index]})
        for action in sources[id]['evaluation_criteria'].get('actions') or []:
            name,args=action['name'],action['arguments']
            if name not in WRITE_TOOLS and name!='transfer_to_human_agents':continue
            result=env.dispatch_tool(thread,name,args)
            if name in WRITE_TOOLS:
                assert result['code']=='CONFIRMATION_REQUIRED',(id,result)
                assert user.decide(result['proposal'])==(True,None),(id,result['proposal'])
                assert env.confirm_action(thread,result['proposal_id'],True)['ok']
                result=env.dispatch_tool(thread,name,args)
                events.append({'type':'confirmed_write','order_id':args['order_id']})
            else:events.append({'type':name})
            assert result['ok'],(id,result)
        # Literal evidence here validates the scorer only; it is explicitly not a model answer.
        events.append({'type':'assistant_message','text':' '.join(map(str,task['expected_checks']['communicate_info']))})
        after=env.load_session(thread)['business']
        checked=check_final_state(task,before,after,events)
        assert checked['passed'],(id,checked)
        assert not check_final_state(task,before,before,[])['passed'],id
        if task['script'].get('required_proposal_rules'):
            assert 'missing_user_revision' in check_final_state(task,before,after,[e for e in events if e['type']!='user_revision'])['failures']


def test_paypal_only_customer_never_authorizes_original_card_fallback():
    from retailflow.user_script import ScriptedUser, user_view
    cfg=json.loads((ROOT/'configs/coverage-evaluation.json').read_text())
    task=next(json.loads(l) for l in (ROOT/cfg['dataset_root']/'data/derived/dev.jsonl').read_text().splitlines() if json.loads(l)['source_task_id']=='12')
    user=ScriptedUser(user_view(task),cfg['user_consent']['tau-retail-12'])
    assert 'PayPal' in user.answer({'fields':['refund_method']})
    assert 'keyboard' in user.answer({'fields':['return_items']})


def test_new_dialogue_fields_count_as_legal_protocol():
    from langchain_core.messages import AIMessage
    from retailflow.evaluation import _tool_stats
    message=AIMessage(content='',tool_calls=[{'name':'ask_user','id':'ask','args':{'fields':['exchange_items','exchange_options','payment_method'],'question':'Which item and payment?'}}])
    stats,_=_tool_stats([message])
    assert stats['legal_tool_requests']==1
