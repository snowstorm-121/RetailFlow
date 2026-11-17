import asyncio,json
from pathlib import Path
from langchain_core.messages import HumanMessage,SystemMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from retailflow.cancel_goals import GOAL_TOOL,GOAL_PROMPT,validate_goal
CASES={
 'negated':(['I do not want to cancel order #W5056519.'], 'none', []),
 'withdraw_polite':(['Cancel all of my pending orders.','Actually, leave everything as it is.'], 'none', []),
 'replace':(['Cancel #W5056519.','Keep #W5056519. Cancel #W5995614 instead.'], 'selected', ['#W5995614']),
 'narrow':(['Cancel #W5056519 and #W5995614.','Only cancel #W5995614; keep #W5056519.'], 'selected', ['#W5995614']),
 'reason':(['Please cancel all pending orders.','My email is yara@example.com.','I no longer need them.'], 'all_pending', []),
 'query':(['What is the status of order #W5056519?'], 'none', []),
 'return':(['Please return the delivered order #W5056519.'], 'none', []),
 'correction_then_reason':(['Cancel all pending orders.','Actually only cancel #W5995614.','I ordered it by mistake.'], 'selected', ['#W5995614']),
}
async def main():
 root=Path('artifacts/goal-stability-20261117');summary=[]
 for label,(messages,scope,ids) in CASES.items():
  result=await load_chat_model(Context(trace_path=str(root/f'{label}-http.jsonl'))).bind_tools([GOAL_TOOL]).ainvoke([SystemMessage(content=GOAL_PROMPT),HumanMessage(content=json.dumps({'customer_messages':messages}))],stream=False)
  calls=result.tool_calls;args=calls[0]['args'] if len(calls)==1 and calls[0]['name']=='update_cancel_goal' else {}
  passed=bool(args and validate_goal(args,messages) and args['scope']==scope and set(args['order_ids'])==set(ids))
  (root/f'{label}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2))
  summary.append({'case':label,'messages':messages,'expected_scope':scope,'expected_ids':ids,'args':args,'passed':passed});print(label,passed,args,repr(result.content),flush=True)
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
asyncio.run(main())
