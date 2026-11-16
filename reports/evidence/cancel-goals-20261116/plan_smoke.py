import asyncio,json
from pathlib import Path
from langchain_core.messages import HumanMessage,SystemMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
TOOL={'type':'function','function':{'name':'update_cancel_goal','description':'Record the current requested cancellation scope. This does not cancel or authorize anything. Use none when cancellation is withdrawn or not requested.','parameters':{'type':'object','properties':{'scope':{'type':'string','enum':['none','all_pending','selected']},'order_ids':{'type':'array','items':{'type':'string'}},'source_quote':{'type':'string'}},'required':['scope','order_ids','source_quote'],'additionalProperties':False}}}
CASES={
 'all':['I want to cancel all of my pending orders.'],
 'reason':['I want to cancel all of my pending orders.','My name is Yara Muller, ZIP 85041.','I ordered them by mistake.'],
 'withdraw':['I want to cancel all of my pending orders.','Actually, keep all my orders. Do not cancel anything.'],
 'narrow':['I want to cancel all of my pending orders.','Actually, cancel only #W5056519 and keep the other orders.'],
 'return':['Please return everything I just bought except the coffee machine.'],
}
async def main():
 root=Path('artifacts/cancel-goals-20261116')
 for label,messages in CASES.items():
  result=await load_chat_model(Context(trace_path=str(root/f'{label}-http.jsonl'))).bind_tools([TOOL]).ainvoke([SystemMessage(content='Read the customer messages in chronological order. Record the latest cancellation intent using one native update_cancel_goal call. Later corrections override earlier scope. Identity or reason answers do not cancel an earlier request. For all_pending or none use an empty order_ids list. For selected use only exact order IDs the customer supplied. source_quote must quote the customer message establishing the latest scope. Never output plain text.'),HumanMessage(content=json.dumps({'customer_messages':messages}))],stream=False)
  (root/f'{label}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2));print(label,repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
