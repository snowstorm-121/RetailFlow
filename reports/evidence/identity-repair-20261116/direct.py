import asyncio,json
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/identity-repair-20261116')
 cases={
 'direct':[SystemMessage(content='Use ask_user to request the missing information. Call the tool; do not answer in text.'),HumanMessage(content='Request the customer email or full name and ZIP.')],
 'no_examples':[SystemMessage(content='Make one native ask_user tool call requesting the customer information in the quoted message. Use the matching allowed field categories. Do not answer the message, do not invent facts, and do not output plain text.'),HumanMessage(content=json.dumps({'customer_information_request':'I need your email or full name and ZIP to locate your account and orders. Could you please provide that information?'}))]}
 for name,messages in cases.items():
  result=await load_chat_model(Context(trace_path=str(root/f'{name}-http.jsonl'))).bind_tools([RETAIL_TOOLS[-1]]).ainvoke(messages,stream=False)
  (root/f'{name}-parsed.json').write_text(json.dumps(result.model_dump(),indent=2)); print(name,repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
