import asyncio,json
from pathlib import Path
from langchain_core.messages import AIMessage, messages_from_dict
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
REMINDER='When you need a tool, use the native tool_calls channel. Never write an action, function, or proposal JSON object in your response text. A JSON object in text does not execute a tool or create a proposal. For a query, answer the requested facts and stop; do not introduce cancellation or return unless the customer requested it.'
async def main():
 root=Path('artifacts/protocol-20261114')
 trace=json.loads(Path('reports/evidence/tool-visibility-20261104/cancel_proposal-baseline-trace.json').read_text())
 messages=messages_from_dict(trace['usage']['context_inputs'][-1]['messages']);messages[0].content="Your previous response described an intended tool action as text, which did not execute. Make one native tool call for that intended action using only verified customer and tool information. Never invent missing fields: use ask_user if necessary. Do not output JSON or confirmation requests as text. The application checks policy and obtains confirmation before writes."
 messages.append(AIMessage(content=json.loads((root/'original-parsed.json').read_text())['content']))
 result=await load_chat_model(Context(trace_path=str(root/'focused-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
 (root/'focused-parsed.json').write_text(json.dumps(result.model_dump(),ensure_ascii=False,indent=2));print(repr(result.content),result.tool_calls)
asyncio.run(main())
