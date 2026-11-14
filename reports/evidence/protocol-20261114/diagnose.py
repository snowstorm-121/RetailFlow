import asyncio,json
from pathlib import Path
from langchain_core.messages import messages_from_dict,SystemMessage
from react_agent.context import Context
from react_agent.utils import load_chat_model
from react_agent.tools import RETAIL_TOOLS

async def main():
    root=Path('artifacts/protocol-20261114')
    trace=json.loads(Path('reports/evidence/tool-visibility-20261104/cancel_proposal-baseline-trace.json').read_text())
    original=messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
    cases={'original':original,'no_examples':[m for m in original if not m.additional_kwargs.get('protocol_example')],
           'minimal':[SystemMessage(content='You are a retail assistant. Use the native cancel_pending_order tool for the requested pending order. Do not output a JSON object as text. The application will ask for confirmation.'),*[m for m in original if m.type!='system' and not m.additional_kwargs.get('protocol_example')]]}
    for name,messages in cases.items():
        model=load_chat_model(Context(trace_path=str(root/f'{name}-http.jsonl'))).bind_tools(RETAIL_TOOLS)
        result=await model.ainvoke(messages,stream=False)
        (root/f'{name}-parsed.json').write_text(json.dumps(result.model_dump(),ensure_ascii=False,indent=2))
        print(name,repr(result.content),result.tool_calls,flush=True)
asyncio.run(main())
