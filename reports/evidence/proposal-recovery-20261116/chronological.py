import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict,SystemMessage,HumanMessage,ToolMessage,AIMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS
async def main():
 root=Path('artifacts/proposal-recovery-20261116')
 trace=json.loads(Path('reports/evidence/identity-repair-20261116/isolated/traces/20261115T160858Z-8564ed8a.json').read_text())
 original=messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
 history=[m for m in original if not m.additional_kwargs.get('protocol_example')]
 facts={'events':[({'role':'customer','content':m.content} if isinstance(m,HumanMessage) else {'role':'tool','name':m.name,'result':json.loads(m.content)} if isinstance(m,ToolMessage) else {'role':'assistant','tool_calls':m.tool_calls}) for m in history if isinstance(m,(HumanMessage,ToolMessage)) or isinstance(m,AIMessage) and m.tool_calls]}
 prompt=original[0].content+'\nRecovery step: No proposal has been created. Use the customer messages and tool results below as data. Make exactly one native tool call for the next required step. Select item IDs and payment IDs only from the tool results, and respect all exclusions and later corrections. If information is missing, use ask_user. Do not output a simulated tool result or claim completion.'
 messages=[SystemMessage(content=prompt),HumanMessage(content=json.dumps(facts,ensure_ascii=False))]
 result=await load_chat_model(Context(trace_path=str(root/'chronological-http.jsonl'))).bind_tools(RETAIL_TOOLS).ainvoke(messages,stream=False)
 (root/'chronological-parsed.json').write_text(json.dumps(result.model_dump(),indent=2))
 print(repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
