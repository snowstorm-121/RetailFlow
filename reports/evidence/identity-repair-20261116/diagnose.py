import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict,SystemMessage,HumanMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS,native_dialogue_examples
async def main():
 root=Path('artifacts/identity-repair-20261116')
 trace=json.loads(Path('artifacts/return-receipt-20261115/dev-B2/traces/20261115T062926Z-3c67b73a.json').read_text())
 original=messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
 question=next(m['data']['content'] for m in trace['messages'] if m['type']=='ai')
 focused=[SystemMessage(content='Make one native ask_user tool call requesting the customer information in the quoted message. Use the matching allowed field categories. Do not answer the message, do not invent facts, and do not output plain text.'),*native_dialogue_examples(),HumanMessage(content=json.dumps({'customer_information_request':question}))]
 for name,messages in [('baseline',original),('focused',focused)]:
  model=load_chat_model(Context(trace_path=str(root/f'{name}-http.jsonl'))).bind_tools([RETAIL_TOOLS[-1]])
  result=await model.ainvoke(messages,stream=False)
  (root/f'{name}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2))
  print(name,repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
