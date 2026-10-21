import asyncio,json,os
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,messages_from_dict,messages_to_dict
from react_agent.context import Context
from react_agent.tools import RETAIL_TOOLS
from react_agent.utils import load_chat_model
os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
root=Path('artifacts/repair-v1');config=json.loads(Path('configs/local.json').read_text())
feedback='Your previous response requests information from the customer, so the task is not complete. Request that information by making one native ask_user call with the allowed field categories. Do not repeat a plain-text question and do not invent customer details.'
async def main():
 results=[]
 original=json.loads((root/'native-diagnostics.json').read_text())[0];failed=json.loads((root/'smoke/result.json').read_text())['outputs'][0];trace=json.loads(Path(failed['trace_path']).read_text())
 inputs=[messages_from_dict(original['input']+original['response']),messages_from_dict(json.loads(Path(failed['trace_path']).read_text())['usage']['context_inputs'][-1]['messages'])+[messages_from_dict(trace['messages'])[-1]]]
 for i,messages in enumerate(inputs):
  messages += [SystemMessage(content=feedback)]
  response=await load_chat_model(Context(**config,trace_path=str(root/f'native-repair-{i}-http.jsonl'))).bind_tools([RETAIL_TOOLS[-1]]).ainvoke(messages,stream=False)
  results.append({'input':messages_to_dict(messages),'response':messages_to_dict([response])});print(i,response.tool_calls,str(response.content)[:160],flush=True)
 (root/'native-repair-diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
asyncio.run(main())
