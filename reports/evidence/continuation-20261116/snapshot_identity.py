import asyncio,json,sqlite3
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/continuation-20261116')
 rows=[json.loads(x) for x in Path('reports/evidence/proposal-recovery-20261116/dev-B2/results.jsonl').read_text().splitlines()]
 r=next(r for r in rows if r['task_id']=='tau-retail-113');trace=json.loads(Path(r['trace_paths'][-1]).read_text())
 original=trace['usage']['context_inputs'][0]['messages'];seen=[];customers=[];outcomes=[]
 for m in original:
  d=m['data']
  if d['additional_kwargs'].get('protocol_example'):continue
  if m['type']=='human':customers.append(d['content'])
  if m['type']=='tool' and d['name']=='get_order_details':
   v=json.loads(d['content'])
   if v.get('ok'):seen.append(v['data']['order_id'])
  if d['additional_kwargs'].get('application_result'):outcomes.append(d['content'])
 with sqlite3.connect('file:artifacts/proposal-recovery-20261116/dev-B2/business.sqlite?mode=ro',uri=True) as db:
  row=db.execute('SELECT business_json, user_id FROM sessions WHERE thread=?',(r['thread_id'],)).fetchone()
  business=json.loads(row[0]); authenticated_user_id=row[1]
 orders=[business['orders'][ident] for ident in dict.fromkeys(seen)]
 prompt=original[0]['data']['content']+'\nThe following are current authoritative snapshots of orders already inspected in this session, after the confirmed action. Review all original requests against these current states. If requested work remains, make the next native tool call. A pending order has not been cancelled. Each new proposal still needs separate application confirmation.'
 messages=[SystemMessage(content=prompt),HumanMessage(content=json.dumps({'authenticated_user_id':authenticated_user_id,'customer_messages':customers,'committed_outcomes':outcomes,'current_observed_orders':orders},ensure_ascii=False))]
 result=await load_chat_model(Context(trace_path=str(root/'snapshot-identity-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
 (root/'snapshot-identity-parsed.json').write_text(json.dumps(result.model_dump(),indent=2));print(repr(result.content),result.tool_calls)
asyncio.run(main())
