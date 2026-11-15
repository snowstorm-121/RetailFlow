"""Opt-in Langfuse tracing; business execution is never retried by telemetry."""
import asyncio
import logging
import os
from contextlib import ExitStack
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _connection():
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    client = Langfuse(public_key=os.environ['LANGFUSE_PUBLIC_KEY'],
                      secret_key=os.environ['LANGFUSE_SECRET_KEY'],
                      base_url=os.environ.get('LANGFUSE_BASE_URL', 'https://jp.cloud.langfuse.com'),
                      timeout=5)
    return client, CallbackHandler


async def trace_turn(thread, event, config, run):
    if os.environ.get('RETAILFLOW_LANGFUSE_ENABLED', '').lower() not in {'true', '1'}:
        return await run(config)
    for key in ('LANGFUSE_SECRET_KEY', 'LANGFUSE_PUBLIC_KEY'):
        if not os.environ.get(key):
            raise ValueError(f'Set {key} locally before enabling Langfuse.')
    stack = ExitStack()
    try:
        from langfuse import propagate_attributes
        client, handler = _connection()
        stack.enter_context(propagate_attributes(session_id=thread, tags=['retailflow'],
                                                 trace_name='retailflow.turn'))
        span = stack.enter_context(client.start_as_current_observation(
            name='retailflow.turn', as_type='agent', input=event))
        callbacks = [handler(public_key=os.environ['LANGFUSE_PUBLIC_KEY'])]
        trace_id = client.get_current_trace_id()
    except Exception:
        try:
            stack.close()
        except Exception:
            pass
        log.warning('Langfuse initialization failed; continuing with local traces.')
        return await run(config)
    try:
        result = await run({**config, '_langfuse_callbacks': callbacks, '_langfuse_trace_id': trace_id})
        try:
            span.update(output={k: result.get(k) for k in ('status', 'text', 'request', 'failure_reason')},
                        metadata={'local_trace_path': result.get('trace_path'),
                                  'tool_attempts': result.get('usage', {}).get('tool_attempts', []),
                                  'confirmed_result': result.get('usage', {}).get('confirmed_result'),
                                  'model_calls': result.get('usage', {}).get('model_calls')},
                        level='ERROR' if result.get('status') in {'failed', 'budget_exceeded'} else 'DEFAULT')
            result['langfuse_trace_id'] = trace_id
        except Exception:
            log.warning('Langfuse trace update failed; business result is preserved.')
        return result
    except Exception:
        try:
            span.update(level='ERROR', status_message='Business execution raised an exception.')
        except Exception:
            pass
        raise
    finally:
        try:
            stack.close()
            await asyncio.to_thread(client.flush)
        except Exception:
            log.warning('Langfuse export failed; local traces remain available.')
