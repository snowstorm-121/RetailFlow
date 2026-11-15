import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('reports/evidence/protocol-20261114/dev-B2')
 row=next(json.loads(l) for l in (root/'results.jsonl').read_text().splitlines() if json.loads(l)['task_id']=='tau-retail-48')
 trace=json.loads((root/'traces'/Path(row['trace_paths'][-1]).name).read_text())
 for variant in ['baseline','receipt_facts']:
  messages=messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
  if variant=='receipt_facts':
   for m in messages:
    if m.additional_kwargs.get('application_result'):
     m.content+='\nRETURN POLICY FACTS: This is a return request, not a completed refund. No refund arrival date is established. No further items in this same order can be returned after this request, including items not selected in it. Do not say the remaining items are eligible for a later return. Answer any additional customer questions using these facts and the tool results.'
  result=await load_chat_model(Context(trace_path=f'artifacts/receipt-20261115/{variant}-http.jsonl')).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
  Path(f'artifacts/receipt-20261115/{variant}.json').write_text(json.dumps(result.model_dump(),ensure_ascii=False,indent=2));print(variant,result.content,flush=True)
asyncio.run(main())
