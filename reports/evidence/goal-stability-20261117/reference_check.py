import asyncio,json,re
from pathlib import Path
from langchain_core.messages import HumanMessage,SystemMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from retailflow.cancel_goals import validate_goal
TOOL={'type':'function','function':{'name':'update_cancel_goal','description':'Record the current requested cancellation scope. This does not cancel or authorize anything. Use none when cancellation is withdrawn or not requested.','parameters':{'type':'object','properties':{'scope':{'type':'string','enum':['none','all_pending','selected']},'order_indices':{'type':'array','items':{'type':'integer'}},'source_message_index':{'type':'integer'}},'required':['scope','order_indices','source_message_index'],'additionalProperties':False}}}
PROMPT='Read the customer messages in chronological order. Record the latest cancellation intent using one native update_cancel_goal call. Later corrections override earlier scope. Identity or reason answers do not cancel an earlier request. For all_pending or none use an empty order_indices list. For selected use zero-based indices from candidate_orders. source_message_index must identify the customer message establishing the latest scope. Never output plain text.'
async def main():
 root=Path('artifacts/goal-stability-20261117');dest=root/'references';dest.mkdir(exist_ok=False);summary=[]
 for case in json.loads((root/'summary.json').read_text()):
  label=case['case'];messages=case['messages'];orders=list(dict.fromkeys(i for m in messages for i in re.findall(r'#W\d+(?!\w)',m)))
  payload={'customer_messages':[{'index':i,'text':m} for i,m in enumerate(messages)],'candidate_orders':[{'index':i,'order_id':o} for i,o in enumerate(orders)]}
  result=await load_chat_model(Context(trace_path=str(dest/f'{label}-http.jsonl'))).bind_tools([TOOL]).ainvoke([SystemMessage(content=PROMPT),HumanMessage(content=json.dumps(payload))],stream=False)
  calls=result.tool_calls;args=calls[0]['args'] if len(calls)==1 and calls[0]['name']=='update_cancel_goal' else {};resolved={}
  if (set(args)=={'scope','order_indices','source_message_index'} and type(args['source_message_index']) is int and 0<=args['source_message_index']<len(messages) and isinstance(args['order_indices'],list) and all(type(i) is int and 0<=i<len(orders) for i in args['order_indices'])):
   resolved={'scope':args['scope'],'order_ids':[orders[i] for i in args['order_indices']],'source_quote':messages[args['source_message_index']]}
  passed=bool(resolved and validate_goal(resolved,messages) and resolved['scope']==case['expected_scope'] and set(resolved['order_ids'])==set(case['expected_ids']))
  (dest/f'{label}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2));summary.append({**case,'args':args,'resolved':resolved,'passed':passed});print(label,passed,args,resolved,flush=True)
 (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
asyncio.run(main())
