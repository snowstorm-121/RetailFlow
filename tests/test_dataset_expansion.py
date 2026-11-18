import asyncio
import json
from copy import deepcopy
from pathlib import Path

from retailflow.dataset import ALLOWED_TOOLS, ROOT, check_final_state, validate_dataset
from retailflow.environment import RetailEnvironment
from retailflow.user_script import ScriptedUser, user_view

CONFIG = ROOT / 'configs/dataset-expansion-v2.json'
SOURCE = ROOT / 'vendor/tau_retail/source/data/tau2/domains/retail'


def test_audit_covers_exactly_the_previously_deferred_tasks():
    source = json.loads((SOURCE/'tasks.json').read_text())
    old = set(json.loads((ROOT/'data/manifests/dataset.json').read_text())['selected_source_ids'])
    expected = {t['id'] for t in source if t['id'] not in old and not
                {a['name'] for a in t['evaluation_criteria'].get('actions') or []} - ALLOWED_TOOLS}
    audit = json.loads((ROOT/'data/reviews/expansion-20261118.json').read_text())['decisions']
    assert len(audit) == len(expected) == 26
    assert {x['id'] for x in audit} == expected
    assert all(x['reason'] for x in audit)


def test_fixed_catalog_selects_supported_fallbacks():
    db = json.loads((SOURCE/'db.json').read_text())
    books = db['products']['8600330539']['variants'].values()
    assert not any(v['available'] and v['options'] == {'material':'glass','color':'white','height':'4 ft'} for v in books)
    keyboards = db['products']['1656367028']['variants'].values()
    cheapest = min((v for v in keyboards if v['available']), key=lambda v:v['price'])
    assert cheapest['price'] == 226.11 and cheapest['price'] >= 200
    assert cheapest['options'] == {'switch type':'tactile','backlight':'white','size':'full size'}
    cameras = db['products']['8940227892']['variants'].values()
    matching = [v for v in cameras if v['available'] and v['options'] == {'resolution':'30MP','zoom':'10x','storage':'SD card'}]
    assert len(matching) == 1 and matching[0]['price'] == 3066.23 > 3000


def test_new_tasks_reference_execution_and_consent(tmp_path):
    config = json.loads(CONFIG.read_text())
    root = ROOT/config['dataset_root']
    assert validate_dataset(str(root/'data/manifests/dataset.json'))['counts'] == {'dev':15,'test':5,'total':20}
    source = {t['id']:t for t in json.loads((SOURCE/'tasks.json').read_text())}
    rows = [json.loads(line) for split in ('dev','test') for line in (root/f'data/derived/{split}.jsonl').read_text().splitlines()]
    env = RetailEnvironment(tmp_path/'states.sqlite')
    for task in rows:
        ident = task['source_task_id']
        if ident not in {'10','11','13','14','88','89','90'}:
            continue
        thread = 'reference-'+ident
        env.create_session(thread, task['initial_state_ref'])
        before = deepcopy(env.load_session(thread)['business'])
        uid = task['group_id'].removeprefix('source-user:')
        assert env.dispatch_tool(thread,'find_user_id_by_email',{'email':before['users'][uid]['email']})['ok']
        user = ScriptedUser(user_view(task),config['user_consent'][task['task_id']])
        events = []
        for action in source[ident]['evaluation_criteria']['actions']:
            result = env.dispatch_tool(thread,action['name'],action['arguments'])
            if result['code']=='CONFIRMATION_REQUIRED':
                proposal = env.pending_proposal(thread)
                assert user.decide(proposal) == (True,None)
                assert env.confirm_action(thread,proposal['proposal_id'],True)['ok']
                result = env.dispatch_tool(thread,action['name'],action['arguments'])
                events.append({'type':'confirmed_write','order_id':action['arguments']['order_id']})
            assert result['ok'], result
            if action['name']=='transfer_to_human_agents':
                events.append({'type':'transfer_to_human_agents'})
        # An oracle witness for the checker, not a model-generated answer or score.
        events.append({'type':'assistant_message','text':' '.join(map(str,task['expected_checks']['communicate_info']))})
        after = env.load_session(thread)['business']
        assert check_final_state(task,before,after,events)['passed']
        if task['expected_checks']['orders']:
            assert not check_final_state(task,before,before,events)['passed']
        else:
            assert not check_final_state(task,before,after,[])['passed']


def test_evaluation_reads_configured_dataset(monkeypatch,tmp_path):
    import retailflow.evaluation as evaluation
    monkeypatch.setattr(evaluation,'build_manifest',lambda *args:{'manifest_sha256':'test-only'})
    async def fake_episode(task,variant,config):
        return {'task_id':task['task_id'],'category':task['category'],'status':'failed','success':False,
                'failure_reason':'test-only','review_required':False}
    monkeypatch.setattr(evaluation,'evaluate_episode',fake_episode)
    result = asyncio.run(evaluation.evaluate_batch('dev','B2',CONFIG,tmp_path/'output'))
    assert result['n_assigned'] == 15
    assigned = json.loads((tmp_path/'output/manifest.json').read_text())['assigned_task_ids']
    assert 'tau-retail-88' in assigned and 'tau-retail-89' in assigned


def test_environment_manifest_hashes_selected_dataset(monkeypatch):
    import retailflow.evaluation as evaluation
    config = json.loads(CONFIG.read_text())
    class Response:
        def __init__(self,value):
            self.value=value
        def json(self):
            return self.value
    def fake_get(url,**kwargs):
        return Response({'models':[{'name':config['execution']['model'],'digest':config['expected_model_digest']}]} if url.endswith('/tags') else {'version':'test-only'})
    monkeypatch.setattr(evaluation.httpx,'get',fake_get)
    hashes=evaluation.build_manifest(CONFIG,config)['file_hashes']
    assert config['dataset_root']+'/data/derived/dev.jsonl' in hashes
    assert config['dataset_root']+'/data/derived/test.jsonl' in hashes
    assert 'data/derived/dev.jsonl' not in hashes
