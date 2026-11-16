import asyncio,json,sqlite3
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from retailflow.environment import RetailEnvironment
from retailflow.tools import WRITE_TOOLS
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/continuation-20261116');target=root/'read-chain.sqlite'
 assert not target.exists()
 with sqlite3.connect('file:artifacts/proposal-recovery-20261116/dev-B2/business.sqlite?mode=ro',uri=True) as source,sqlite3.connect(target) as dest: source.backup(dest)
 r=next(json.loads(x) for x in Path('reports/evidence/proposal-recovery-20261116/dev-B2/results.jsonl').read_text().splitlines() if json.loads(x)['task_id']=='tau-retail-113')
 env=RetailEnvironment(target);before=env.state_hash(r['thread_id'])
 req=json.loads((root/'snapshot-identity-http.jsonl').read_text().splitlines()[0])['body']
 messages=[SystemMessage(content=req['messages'][0]['content']),HumanMessage(content=req['messages'][1]['content'])]
 response=AIMessage(**json.loads((root/'snapshot-identity-parsed.json').read_text()));steps=[]
 for i in range(4):
  messages.append(response);calls=response.tool_calls
  if len(calls)!=1:break
  call=calls[0];steps.append({'call':call})
  if call['name'] in WRITE_TOOLS or call['name'] in ('ask_user','transfer_to_human_agents'):break
  outcome=env.dispatch_tool(r['thread_id'],call['name'],call['args']);steps[-1]['result']=outcome
  messages.append(ToolMessage(content=json.dumps(outcome),tool_call_id=call['id'],name=call['name']))
  response=await load_chat_model(Context(trace_path=str(root/'read-chain-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
  (root/f'read-chain-{i}-parsed.json').write_text(json.dumps(response.model_dump(),indent=2))
  print(i,repr(response.content),response.tool_calls,flush=True)
 assert env.state_hash(r['thread_id'])==before
 (root/'read-chain-summary.json').write_text(json.dumps({'steps':steps,'business_unchanged':True,'final':response.model_dump()},indent=2))
asyncio.run(main())
