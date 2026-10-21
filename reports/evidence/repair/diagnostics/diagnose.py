import asyncio,json,os
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage,messages_to_dict
from react_agent.context import Context
from react_agent.tools import RETAIL_TOOLS
from react_agent.utils import load_chat_model
from retailflow.runtime import RETAIL_PROMPT
os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
root=Path('artifacts/repair-v1');config=json.loads(Path('configs/local.json').read_text())
short='You help retail customers using tools. If identity is missing, call ask_user(fields=["identity"], question="What is your email or full name and ZIP?"). Use native tool calls, not text questions. Never invent identity or IDs. Use one tool per response and empty content when calling it. Only answer when the customer request is complete.'
async def main():
 results=[]
 for variant in ['original','concise','native_example']:
  for i,opening in enumerate(['I want to return a damaged item.','Please cancel my order.']):
   prompt=short if variant=='concise' else RETAIL_PROMPT.format(system_time='2026-10-21T00:00:00+08:00')
   prefix=[SystemMessage(content=prompt)]
   if variant=='native_example':
    args={'fields':['identity'],'question':'What is your email or full name and ZIP?'}
    prefix += [HumanMessage(content='I need help with my purchase.'),AIMessage(content='',tool_calls=[{'name':'ask_user','args':args,'id':'example-ask'}]),ToolMessage(content=json.dumps({'ok':True,'code':'NEEDS_INPUT','data':args}),name='ask_user',tool_call_id='example-ask')]
   messages=[*prefix,HumanMessage(content=opening)]
   model=load_chat_model(Context(**config,trace_path=str(root/f'{variant}-{i}-http.jsonl'))).bind_tools(RETAIL_TOOLS)
   response=await model.ainvoke(messages,stream=False)
   result={'variant':variant,'opening':opening,'input':messages_to_dict(messages),'response':messages_to_dict([response]),'content':response.content,'tool_calls':response.tool_calls,'usage':response.usage_metadata}
   results.append(result);(root/'native-diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
   print(variant,i,'tools=',response.tool_calls,'content=',str(response.content)[:220],flush=True)
asyncio.run(main())
