import asyncio,json
from pathlib import Path
from langchain_core.messages import SystemMessage,HumanMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
async def main():
 root=Path('artifacts/receipt-20261115')
 draft=json.loads((root/'baseline.json').read_text())['content']
 messages=[SystemMessage(content='Correct the draft using the verified facts. Preserve its answers to additional customer questions, correcting them where they contradict the facts. Do not invent dates or actions. Output only the corrected customer-facing answer.'),HumanMessage(content=json.dumps({'draft':draft,'verified_facts':{'status':'return requested','refund_status':'not settled; no arrival date established','additional_returns_for_same_order':'not allowed, including remaining items','email':'no real email is sent by this local application'}}))]
 r=await load_chat_model(Context(trace_path=str(root/'correction-http.jsonl'))).ainvoke(messages,stream=False)
 (root/'correction.json').write_text(json.dumps(r.model_dump(),ensure_ascii=False,indent=2));print(r.content)
asyncio.run(main())
