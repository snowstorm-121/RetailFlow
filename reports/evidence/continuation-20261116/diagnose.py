import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict,SystemMessage,HumanMessage,ToolMessage,AIMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/continuation-20261116')
 rows=[json.loads(x) for x in Path('reports/evidence/proposal-recovery-20261116/dev-B2/results.jsonl').read_text().splitlines()]
 for task in ('tau-retail-113','tau-retail-2'):
  row=next(r for r in rows if r['task_id']==task)
  trace=json.loads(Path(row['trace_paths'][-1]).read_text());original=messages_from_dict(trace['usage']['context_inputs'][0]['messages'])
  events=[]
  for m in original:
   if m.additional_kwargs.get('protocol_example'):continue
   if isinstance(m,HumanMessage):events.append({'role':'customer','content':m.content})
   elif isinstance(m,ToolMessage):events.append({'role':'tool','name':m.name,'result':json.loads(m.content)})
   elif isinstance(m,AIMessage) and m.tool_calls:events.append({'role':'assistant','tool_calls':m.tool_calls})
   elif m.additional_kwargs.get('application_result'):events.append({'role':'application','content':m.content})
  prompt=original[0].content+'\nContinuation step: The application outcome records only the displayed confirmed action. Review all customer requests and later corrections against the recorded outcomes. Continue with a native tool call for any remaining requested action or unanswered factual question. Do not ask the customer to repeat an existing request. Never repeat a committed action or assume confirmation of another proposal. If every requested goal is complete, give an accurate final answer.'
  messages=[SystemMessage(content=prompt),HumanMessage(content=json.dumps({'events':events},ensure_ascii=False))]
  result=await load_chat_model(Context(trace_path=str(root/f'{task}-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
  (root/f'{task}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2));print(task,repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
