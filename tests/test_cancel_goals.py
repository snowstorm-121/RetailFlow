from retailflow.cancel_goals import validate_goal, goal_progress


def test_goal_rejects_empty_selected_and_fabricated_quote():
    assert not validate_goal({'scope':'selected','order_ids':[], 'source_quote':'Return my order.'}, ['Return my order.'])
    assert not validate_goal({'scope':'all_pending','order_ids':[], 'source_quote':'Cancel all.'}, ['Keep everything.'])
    assert not validate_goal({'scope':'selected','order_ids':['#W2'], 'source_quote':'Cancel #W1.'}, ['Cancel #W1.'])


def test_goal_keeps_withdrawal_and_selected_scope():
    assert validate_goal({'scope':'none','order_ids':[], 'source_quote':'Keep everything.'}, ['Cancel all.', 'Keep everything.'])
    assert validate_goal({'scope':'selected','order_ids':['#W1'], 'source_quote':'Only cancel #W1.'}, ['Cancel all.', 'Only cancel #W1.'])


def test_progress_needs_inventory_and_commit_evidence():
    goal={'scope':'all_pending','order_ids':[], 'targets':None}
    business={'orders':{'#W1':{'user_id':'u','status':'pending'}, '#W2':{'user_id':'other','status':'pending'}}}
    assert not goal_progress(goal,'u',business,None,set())['complete']
    p=goal_progress(goal,'u',business,['#W1','#W2'],set())
    assert p['remaining_order_ids']==['#W1']
    business['orders']['#W1']['status']='cancelled'
    assert not goal_progress(goal,'u',business,['#W1'],set())['complete']
    assert goal_progress(goal,'u',business,['#W1'],{'#W1'})['complete']
    assert goal['targets']==['#W1']


def test_selected_foreign_order_does_not_leak_status_or_count_complete():
    goal={'scope':'selected','order_ids':['#W2'],'targets':None}
    p=goal_progress(goal,'u',{'orders':{'#W2':{'user_id':'other','status':'cancelled'}}},None,{'#W2'})
    assert not p['complete']
    assert p['orders']==[{'order_id':'#W2','status':'unavailable','committed':False}]


def test_runtime_registers_persists_and_blocks_unfinished_goal(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime, call
    opening='Yara Muller, 85041. Cancel all of my pending orders.'
    worker, model, config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'all_pending','order_ids':[],'source_quote':opening},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_user_details',{'user_id':'yara_muller_8652'},'profile'),
        AIMessage(content='All orders cancelled.'), AIMessage(content='All done.'),
    ], cancel_goal_tracking=True, isolated_proposal_repair=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':opening},config))
    assert result['status']=='failed'
    assert result['failure_reason']=='cancel_goal_incomplete'
    goal=worker._load_runtime('a')['cancel_goal']
    assert set(goal['targets'])=={'#W5056519','#W5995614'}
    assert not goal['needs_refresh']
    assert worker.environment.get_order('a','#W5056519')['status']=='pending'
    assert 'cancel_goal' in model.inputs[-1][1].content


def test_invalid_native_plan_stops_before_business(tmp_path):
    import asyncio
    from test_agent_loop import runtime, call
    worker,model,config=runtime(tmp_path,[call('update_cancel_goal',{'scope':'selected','order_ids':[],'source_quote':'Return everything.'},'bad')],cancel_goal_tracking=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':'Return everything.'},config))
    assert result['status']=='failed'
    assert result['failure_reason']=='invalid_cancel_goal'
    assert result['usage']['tool_calls']==1
    assert len(model.inputs)==1


def test_planning_tool_accounting_requires_valid_grounded_plan():
    from langchain_core.messages import HumanMessage
    from retailflow.evaluation import _tool_stats
    from test_agent_loop import call
    text='Cancel all my pending orders.'
    messages=[HumanMessage(content=text),call('update_cancel_goal',{'scope':'all_pending','order_ids':[],'source_quote':text},'g')]
    assert _tool_stats(messages)[0]['legal_tool_requests']==1
    messages[-1]=call('update_cancel_goal',{'scope':'selected','order_ids':[],'source_quote':text},'bad')
    assert _tool_stats(messages)[0]['legal_tool_requests']==0


def test_goal_survives_restart_and_requires_each_confirmation(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from retailflow.runtime import RetailRuntime
    from test_agent_loop import runtime, call
    text='Yara Muller, 85041. Cancel all my pending orders; ordered by mistake.'
    worker,model,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'all_pending','order_ids':[],'source_quote':text},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_user_details',{'user_id':'yara_muller_8652'},'profile'),
        call('get_order_details',{'order_id':'#W5056519'},'o1'),
        call('cancel_pending_order',{'order_id':'#W5056519','reason':'ordered by mistake'},'c1'),
        AIMessage(content='All cancelled.'),
        call('get_order_details',{'order_id':'#W5995614'},'o2'),
        call('cancel_pending_order',{'order_id':'#W5995614','reason':'ordered by mistake'},'c2'),
        AIMessage(content='Both orders cancelled.'),
    ],cancel_goal_tracking=True,isolated_proposal_repair=True)
    first=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert first['status']=='needs_confirmation'
    worker=RetailRuntime(tmp_path/'business.sqlite',tmp_path/'traces',model_factory=lambda _:model)
    second=asyncio.run(worker.run_turn('a',{'kind':'confirmation','accepted':True,'proposal_id':first['request']['proposal_id']},config))
    assert second['status']=='needs_confirmation'
    assert second['request']['args']['order_id']=='#W5995614'
    assert worker.environment.get_order('a','#W5056519')['status']=='cancelled'
    assert worker.environment.get_order('a','#W5995614')['status']=='pending'
    final=asyncio.run(worker.run_turn('a',{'kind':'confirmation','accepted':True,'proposal_id':second['request']['proposal_id']},config))
    assert final['status']=='answered'
    assert worker.environment.get_order('a','#W5995614')['status']=='cancelled'


def test_customer_withdrawal_invalidates_goal_and_old_proposal(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime, call
    text='Yara Muller, 85041. Cancel #W5056519; ordered by mistake.'
    correction='Actually, keep all my orders. Do not cancel anything.'
    worker,_,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'selected','order_ids':['#W5056519'],'source_quote':text},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_order_details',{'order_id':'#W5056519'},'o'),
        call('cancel_pending_order',{'order_id':'#W5056519','reason':'ordered by mistake'},'c'),
        call('update_cancel_goal',{'scope':'none','order_ids':[],'source_quote':correction},'withdraw'),
        AIMessage(content='I will keep your orders.'),
    ],cancel_goal_tracking=True)
    proposal=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':correction},config))
    assert result['status']=='answered'
    assert worker._load_runtime('a')['cancel_goal']['scope']=='none'
    stale=asyncio.run(worker.run_turn('a',{'kind':'confirmation','accepted':True,'proposal_id':proposal['request']['proposal_id']},config))
    assert stale['status']=='failed'
    assert worker.environment.get_order('a','#W5056519')['status']=='pending'


def test_none_goal_rejects_model_cancel_proposal(tmp_path):
    import asyncio
    from test_agent_loop import runtime, call
    text='Yara Muller, 85041. Keep all orders; do not cancel anything.'
    worker,_,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'none','order_ids':[],'source_quote':text},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_order_details',{'order_id':'#W5056519'},'o'),
        call('cancel_pending_order',{'order_id':'#W5056519','reason':'ordered by mistake'},'bad'),
    ],cancel_goal_tracking=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert result['status']=='refused'
    assert worker.environment.pending_proposal('a') is None


def test_selected_goal_requires_exact_order_id_in_quote():
    assert not validate_goal({'scope':'selected','order_ids':['#W1'],'source_quote':'Cancel #W12.'}, ['Cancel #W12.'])


def test_goal_cannot_finish_before_identity_or_inventory(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime, call
    text='Cancel all my pending orders.'
    worker,_,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'all_pending','order_ids':[],'source_quote':text},'goal'),
        AIMessage(content='All done.'), AIMessage(content='All done.'),
    ],cancel_goal_tracking=True,isolated_proposal_repair=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert result['status']=='failed'
    assert result['failure_reason']=='cancel_goal_incomplete'


def test_incomplete_goal_still_allows_missing_reason_question(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime, call
    text='Yara Muller, 85041. Cancel all my pending orders.'
    worker,model,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'all_pending','order_ids':[],'source_quote':text},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_user_details',{'user_id':'yara_muller_8652'},'profile'),
        AIMessage(content='Please provide the cancellation reason.'),
        call('ask_user',{'fields':['cancel_reason'],'question':'What is the cancellation reason?'},'ask'),
    ],cancel_goal_tracking=True,isolated_information_repair=True,isolated_proposal_repair=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert result['status']=='needs_input'
    assert result['request']['fields']==['cancel_reason']
    assert worker.environment.pending_proposal('a') is None
    assert [t['function']['name'] for t in model.bound_tools]==['ask_user']


def test_unchanged_goal_refresh_preserves_frozen_unproven_targets(tmp_path):
    import asyncio,json
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime,call
    text='Yara Muller, 85041. Cancel all my pending orders.'
    goal={'scope':'all_pending','order_ids':[],'source_quote':text}
    worker,_,config=runtime(tmp_path,[
        call('update_cancel_goal',goal,'g1'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_user_details',{'user_id':'yara_muller_8652'},'profile'),
        call('ask_user',{'fields':['cancel_reason'],'question':'Reason?'},'ask'),
        call('update_cancel_goal',goal,'g2'),
        AIMessage(content='All done.'),AIMessage(content='All done.'),
    ],cancel_goal_tracking=True,isolated_proposal_repair=True)
    first=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert first['status']=='needs_input'
    targets=worker._load_runtime('a')['cancel_goal']['targets']
    # Deliberately corrupt only this test database: changed status is not a receipt.
    with worker.environment.storage.transaction() as db:
        business=json.loads(db.execute('SELECT business_json FROM sessions WHERE thread=?',('a',)).fetchone()[0])
        for ident in targets:business['orders'][ident]['status']='cancelled'
        db.execute('UPDATE sessions SET business_json=? WHERE thread=?',(json.dumps(business),'a'))
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':'Ordered by mistake.'},config))
    assert worker._load_runtime('a')['cancel_goal']['targets']==targets
    assert result['status']=='failed'
    assert result['failure_reason']=='cancel_goal_incomplete'


def test_cancel_goal_rejects_explicit_keep_and_return_only_evidence():
    assert not validate_goal({'scope':'selected','order_ids':['#W5056519'],'source_quote':'Keep #W5056519. Cancel #W5995614 instead.'}, ['Keep #W5056519. Cancel #W5995614 instead.'])
    assert not validate_goal({'scope':'selected','order_ids':['#W5056519'],'source_quote':'Please return order #W5056519.'}, ['Please return order #W5056519.'])
    assert not validate_goal({'scope':'all_pending','order_ids':[],'source_quote':'Cancel all pending orders.'}, ['Cancel all pending orders.','Keep #W5056519.'])
    assert validate_goal({'scope':'selected','order_ids':['#W5995614'],'source_quote':'Keep #W5056519. Cancel #W5995614 instead.'}, ['Keep #W5056519. Cancel #W5995614 instead.'])


def test_goal_reference_resolution_uses_original_ids_and_rejects_invalid_indices():
    from retailflow.cancel_goals import resolve_goal_references
    messages=['Cancel #W5056519.','Keep #W5056519. Cancel #W5995614 instead.']
    assert resolve_goal_references({'scope':'selected','order_indices':[1],'source_message_index':1},messages)=={'scope':'selected','order_ids':['#W5995614'],'source_quote':messages[1]}
    for bad in (-1,2,True,'1'):
        assert resolve_goal_references({'scope':'selected','order_indices':[bad],'source_message_index':1},messages) is None
    assert resolve_goal_references({'scope':'selected','order_indices':[1,1],'source_message_index':1},messages) is None
    assert resolve_goal_references({'scope':'selected','order_indices':[0],'source_message_index':1},messages) is None


def test_reference_goal_runtime_stores_resolved_original_values(tmp_path):
    import asyncio
    from test_agent_loop import runtime,call
    text='Yara Muller, 85041. Cancel #W5056519; ordered by mistake.'
    worker,model,config=runtime(tmp_path,[
        call('update_cancel_goal',{'scope':'selected','order_indices':[0],'source_message_index':0},'goal'),
        call('find_user_id_by_name_zip',{'first_name':'Yara','last_name':'Muller','zip':'85041'},'id'),
        call('get_order_details',{'order_id':'#W5056519'},'o'),
        call('cancel_pending_order',{'order_id':'#W5056519','reason':'ordered by mistake'},'c'),
    ],cancel_goal_tracking=True,cancel_goal_references=True)
    result=asyncio.run(worker.run_turn('a',{'kind':'message','text':text},config))
    assert result['status']=='needs_confirmation'
    goal=worker._load_runtime('a')['cancel_goal']
    assert goal['order_ids']==['#W5056519'] and goal['source_quote']==text
    assert 'candidate_orders' in model.inputs[0][1].content


def test_goal_resolution_survives_model_context_filter():
    import json
    from langchain_core.messages import ToolMessage
    from retailflow.context import _tool_view
    data={'scope':'selected','order_ids':['#W5056519'],'source_quote':'Cancel #W5056519.'}
    message=ToolMessage(content=json.dumps({'ok':True,'code':'GOAL_RECORDED','data':data}),name='update_cancel_goal',tool_call_id='g')
    assert json.loads(_tool_view(message).content)['data']==data


def test_reference_source_index_and_legal_call_accounting():
    from retailflow.cancel_goals import resolve_goal_references
    from retailflow.evaluation import _tool_stats
    from langchain_core.messages import HumanMessage
    from test_agent_loop import call
    text='Cancel #W5056519.'
    for source in (-1,1,True,'0'):
        assert resolve_goal_references({'scope':'selected','order_indices':[0],'source_message_index':source},[text]) is None
    valid=call('update_cancel_goal',{'scope':'selected','order_indices':[0],'source_message_index':0},'goal')
    assert _tool_stats([HumanMessage(content=text),valid])[0]['legal_tool_requests']==1


import pytest


@pytest.mark.parametrize('text', [
    'I want to cancel all of my pending orders.',
    'Cancel all my pending orders',
    'Please cancel all pending orders.',
    "I'd like to cancel all of my pending orders!",
    '  CANCEL ALL MY PENDING ORDERS.  ',
])
def test_explicit_all_pending_activation(text):
    from retailflow.cancel_goals import is_explicit_all_pending_request
    assert is_explicit_all_pending_request(text)


@pytest.mark.parametrize('text', [
    "Don't cancel all my pending orders.",
    'Can I cancel all my pending orders?',
    'If possible, cancel all my pending orders.',
    'Cancel all my pending orders except #W1.',
    'Cancel all my pending orders. Actually keep #W1.',
    'Cancel all my pending orders and return my laptop.',
    'Please undo an order I cancelled.',
    'Return everything except the coffee machine.',
    'Cancel the grill I ordered.',
    'Cancel #W1.',
    'What does "cancel all my pending orders" mean?',
    '',
])
def test_activation_leaves_other_or_ambiguous_requests_on_normal_path(text):
    from retailflow.cancel_goals import is_explicit_all_pending_request
    assert not is_explicit_all_pending_request(text)


def test_scoped_goal_does_not_intercept_other_business(tmp_path):
    import asyncio
    from test_agent_loop import runtime, call
    worker, model, config = runtime(tmp_path, [
        call('ask_user', {'fields': ['identity'], 'question': 'Please give your name and ZIP.'}, 'ask'),
    ], cancel_goal_tracking=True, cancel_goal_activation='explicit_all_pending')
    result = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Return my laptop.'}, config))
    assert result['status'] == 'needs_input'
    assert 'cancel_goal' not in worker._load_runtime('a')
    assert 'update_cancel_goal' not in [t['function']['name'] for t in model.bound_tools]


def test_scoped_goal_stays_active_for_withdrawal_after_restart(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from retailflow.runtime import RetailRuntime
    from test_agent_loop import runtime, call
    text = 'Please cancel all pending orders.'
    worker, model, config = runtime(tmp_path, [
        call('update_cancel_goal', {'scope': 'all_pending', 'order_ids': [], 'source_quote': text}, 'goal'),
        call('ask_user', {'fields': ['identity'], 'question': 'Name and ZIP?'}, 'ask'),
        call('update_cancel_goal', {'scope': 'none', 'order_ids': [], 'source_quote': 'Keep everything.'}, 'withdraw'),
        AIMessage(content='No orders were cancelled.'),
    ], cancel_goal_tracking=True, cancel_goal_activation='explicit_all_pending')
    first = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': text}, config))
    assert first['status'] == 'needs_input'
    worker = RetailRuntime(tmp_path/'business.sqlite', tmp_path/'traces', model_factory=lambda _: model)
    second = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': 'Keep everything.'}, config))
    assert second['status'] == 'answered'
    assert worker._load_runtime('a')['cancel_goal']['scope'] == 'none'
    assert worker.environment.pending_proposal('a') is None


def test_scoped_goal_accepts_selected_revision_and_confirmation(tmp_path):
    import asyncio
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime, call
    opening = 'Cancel all my pending orders.'
    revision = 'Yara Muller, 85041. Only cancel #W5056519, ordered by mistake.'
    worker, _, config = runtime(tmp_path, [
        call('update_cancel_goal', {'scope': 'all_pending', 'order_ids': [], 'source_quote': opening}, 'goal'),
        call('ask_user', {'fields': ['identity'], 'question': 'Name and ZIP?'}, 'ask'),
        call('update_cancel_goal', {'scope': 'selected', 'order_ids': ['#W5056519'], 'source_quote': revision}, 'revise'),
        call('find_user_id_by_name_zip', {'first_name': 'Yara', 'last_name': 'Muller', 'zip': '85041'}, 'identity'),
        call('get_order_details', {'order_id': '#W5056519'}, 'read'),
        call('cancel_pending_order', {'order_id': '#W5056519', 'reason': 'ordered by mistake'}, 'cancel'),
        AIMessage(content='The selected order was cancelled.'),
    ], cancel_goal_tracking=True, cancel_goal_activation='explicit_all_pending')
    assert asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': opening}, config))['status'] == 'needs_input'
    proposal = asyncio.run(worker.run_turn('a', {'kind': 'message', 'text': revision}, config))
    assert proposal['status'] == 'needs_confirmation'
    assert worker.environment.get_order('a', '#W5056519')['status'] == 'pending'
    result = asyncio.run(worker.run_turn('a', {'kind': 'confirmation', 'accepted': True,
        'proposal_id': proposal['request']['proposal_id']}, config))
    assert result['status'] == 'answered'
    assert worker.environment.get_order('a', '#W5056519')['status'] == 'cancelled'
    assert worker.environment.get_order('a', '#W5995614')['status'] == 'pending'
    assert worker._load_runtime('a')['cancel_goal']['scope'] == 'selected'
