"""Replay a seen Dev failure; never execute returned tools."""
import asyncio
import json
from pathlib import Path
from langchain_core.messages import AIMessage, messages_from_dict
from react_agent.context import Context
from react_agent.tools import RETAIL_TOOLS
from react_agent.utils import load_chat_model

async def main():
    root = Path('artifacts/proposal-context-20261116')
    trace = json.loads(Path('reports/evidence/identity-repair-20261116/isolated/traces/20261115T160858Z-8564ed8a.json').read_text())
    original = messages_from_dict(trace['usage']['context_inputs'][-1]['messages'])
    cases = {
        'original': original,
        'no_examples': [m for m in original if not m.additional_kwargs.get('protocol_example')],
        'no_drafts': [m for m in original if not (isinstance(m, AIMessage) and not m.tool_calls)],
    }
    for name, messages in cases.items():
        target = root / f'{name}-parsed.json'
        if target.exists() or (root / f'{name}-http.jsonl').exists():
            raise FileExistsError(f'Preserve prior evidence: {name}')
        model = load_chat_model(Context(trace_path=str(root / f'{name}-http.jsonl'))).bind_tools(RETAIL_TOOLS)
        result = await model.ainvoke(messages, stream=False)
        target.write_text(json.dumps(result.model_dump(), indent=2))
        print(name, repr(result.content), result.tool_calls, flush=True)

if __name__ == '__main__':
    asyncio.run(main())
