import asyncio,json,os,copy
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage,messages_to_dict
from react_agent.context import Context
from react_agent.tools import RETAIL_TOOLS
from react_agent.utils import load_chat_model
from retailflow.runtime import RETAIL_PROMPT
os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
root=Path('artifacts/repair-v1');config=json.loads(Path('configs/local.json').read_text())
async def main():
 results=[]
 cases=[('contract_only','I want to return a damaged item.'),('contract_only','Please cancel my order.'),('contract_only','I am Yara Muller, ZIP 85041. Cancel #W5056519 because I ordered it by mistake.'),('contract_and_examples','I am Yara Muller, ZIP 85041. Cancel #W5056519 because I ordered it by mistake.')]
 for i,(variant,opening) in enumerate(cases):
  tools=copy.deepcopy(RETAIL_TOOLS);ask=tools[-1]['function'];ask['description']='Request missing customer information. Use this tool instead of asking a question in normal text. Pause and wait for the customer. fields must use the listed categories.';ask['parameters']['properties']['fields'].update(minItems=1,items={'type':'string','enum':['identity','order','cancel_reason','return_items','refund_method']})
  prefix=[SystemMessage(content=RETAIL_PROMPT.format(system_time='2026-10-21T00:00:00+08:00'))]
  if variant=='contract_and_examples':
   for j,example in enumerate(['I want to return something.','Cancel my purchase please.']):
    args={'fields':['identity'],'question':'What is your email or full name and ZIP?'};ident=f'example-ask-{j}'
    prefix += [HumanMessage(content=example),AIMessage(content='',tool_calls=[{'name':'ask_user','args':args,'id':ident}]),ToolMessage(content=json.dumps({'ok':True,'code':'NEEDS_INPUT','data':args}),name='ask_user',tool_call_id=ident)]
  messages=[*prefix,HumanMessage(content=opening)]
  response=await load_chat_model(Context(**config,trace_path=str(root/f'contract-{i}-http.jsonl'))).bind_tools(tools).ainvoke(messages,stream=False)
  results.append({'variant':variant,'opening':opening,'input':messages_to_dict(messages),'response':messages_to_dict([response]),'tools':tools});(root/'contract-diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
  print(variant,i,'tools=',response.tool_calls,'content=',str(response.content)[:160],flush=True)
asyncio.run(main())
