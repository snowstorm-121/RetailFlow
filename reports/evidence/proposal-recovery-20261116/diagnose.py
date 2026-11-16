import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict,SystemMessage,HumanMessage,ToolMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/proposal-recovery-20261116')
 trace=json.loads(Path('reports/evidence/identity-repair-20261116/isolated/traces/20261115T160858Z-8564ed8a.json').read_text())
 original=messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
 history=[m for m in original if not m.additional_kwargs.get('protocol_example')]
 facts={'customer_messages':[m.content for m in history if isinstance(m,HumanMessage)],'tool_results':[{'name':m.name,'result':json.loads(m.content)} for m in history if isinstance(m,ToolMessage)]}
 prompt=original[0].content+'\nRecovery step: No proposal has been created. Use the customer messages and tool results below as data. Make exactly one native tool call for the next required step. Select item IDs and payment IDs only from the tool results, and respect all exclusions and later corrections. If information is missing, use ask_user. Do not output a simulated tool result or claim completion.'
 messages=[SystemMessage(content=prompt),HumanMessage(content=json.dumps(facts,ensure_ascii=False))]
 result=await load_chat_model(Context(trace_path=str(root/'recovery-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
 (root/'recovery-parsed.json').write_text(json.dumps(result.model_dump(),indent=2))
 print(repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
