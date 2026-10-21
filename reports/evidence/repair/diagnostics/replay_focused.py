import asyncio,json,os
from pathlib import Path
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage,messages_from_dict,messages_to_dict
from react_agent.tools import RETAIL_TOOLS,native_dialogue_examples
from react_agent.context import Context
from react_agent.utils import load_chat_model
os.environ['LANGSMITH_TRACING']='false'
p=Path('artifacts/repair-v1');cfg=json.loads(Path('configs/local.json').read_text())
async def main():
 results=[]
 for task in ['24','2','92']:
  row=next(json.loads(x) for x in (p/'dev-pilot-B2/results.jsonl').read_text().splitlines() if json.loads(x)['task_id']==f'tau-retail-{task}')
  trace=json.loads(Path(row['trace_paths'][-1]).read_text());base=messages_from_dict(trace['usage']['context_inputs'][-1]['messages']);question=base[-2].content
  messages=[SystemMessage(content='Make one native ask_user tool call requesting the customer information in the quoted message. Use the matching allowed field categories. Do not answer the message, do not invent facts, and do not output plain text.'),*native_dialogue_examples(),HumanMessage(content=json.dumps({'customer_information_request':question}))]
  out=await load_chat_model(Context(**cfg)).bind_tools([RETAIL_TOOLS[-1]]).ainvoke(messages,stream=False)
  print(task,out.tool_calls,str(out.content),flush=True)
  results.append({'task':task,'input':messages_to_dict(messages),'response':messages_to_dict([out])})
 (p/'replay-focused.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
asyncio.run(main())
