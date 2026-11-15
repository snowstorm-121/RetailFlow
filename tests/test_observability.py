import asyncio
import pytest


def test_disabled_does_not_initialize_sdk(monkeypatch):
    from retailflow import observability as obs
    monkeypatch.delenv('RETAILFLOW_LANGFUSE_ENABLED', raising=False)
    monkeypatch.setattr(obs, '_connection', lambda: pytest.fail('SDK initialized'))
    async def run(config):
        assert '_langfuse_callbacks' not in config
        return {'status': 'answered'}
    assert asyncio.run(obs.trace_turn('a', {'kind': 'message'}, {}, run)) == {'status': 'answered'}


def test_missing_credentials_fail_before_business(monkeypatch):
    from retailflow import observability as obs
    monkeypatch.setenv('RETAILFLOW_LANGFUSE_ENABLED', 'true')
    monkeypatch.delenv('LANGFUSE_SECRET_KEY', raising=False)
    async def run(config):
        pytest.fail('business ran without valid tracing configuration')
    with pytest.raises(ValueError, match='LANGFUSE_SECRET_KEY'):
        asyncio.run(obs.trace_turn('a', {}, {}, run))


def test_setup_failure_runs_business_once_without_tracing(monkeypatch):
    from retailflow import observability as obs
    monkeypatch.setenv('RETAILFLOW_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'secret-test')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-test')
    def broken():
        raise RuntimeError('backend failed')
    monkeypatch.setattr(obs, '_connection', broken)
    calls=[]
    async def run(config):
        calls.append(config)
        return {'status': 'answered'}
    assert asyncio.run(obs.trace_turn('a', {}, {}, run))['status'] == 'answered'
    assert len(calls)==1


def test_real_sdk_exports_graph_and_session_without_network(tmp_path, monkeypatch):
    langfuse = pytest.importorskip('langfuse')
    from langfuse.langchain import CallbackHandler
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from langchain_core.messages import AIMessage
    from test_agent_loop import runtime
    from retailflow import observability as obs

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(public_key='pk-lf-offline-test', secret_key='test-only',
        base_url='https://jp.cloud.langfuse.com', tracer_provider=TracerProvider(), span_exporter=exporter)
    monkeypatch.setenv('RETAILFLOW_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-offline-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'test-only')
    monkeypatch.setattr(obs, '_connection', lambda: (client, CallbackHandler))
    worker, _, config = runtime(tmp_path, [AIMessage(content='Hello.')])
    result = asyncio.run(worker.run_turn('session-offline', {'kind': 'message', 'text': 'Hello'}, config))
    assert result['status'] == 'answered'
    spans = exporter.get_finished_spans()
    root = next(s for s in spans if s.name == 'retailflow.turn')
    assert root.attributes['session.id'] == 'session-offline'
    assert any(s.parent and s.context.trace_id == root.context.trace_id for s in spans)
    assert result['langfuse_trace_id'] == format(root.context.trace_id, '032x')
    client.shutdown()


@pytest.mark.parametrize('business_error', [False, True])
def test_export_failure_never_replays_business(monkeypatch, business_error):
    pytest.importorskip('langfuse')
    from contextlib import nullcontext
    from types import SimpleNamespace
    from retailflow import observability as obs
    monkeypatch.setenv('RETAILFLOW_LANGFUSE_ENABLED', 'true')
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-test')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'secret-test')
    def fail_flush():
        raise RuntimeError('offline')
    span = SimpleNamespace(update=lambda **kw: None)
    client = SimpleNamespace(start_as_current_observation=lambda **kw: nullcontext(span),
                             get_current_trace_id=lambda: 'a'*32, flush=fail_flush)
    monkeypatch.setattr(obs, '_connection', lambda: (client, lambda **kw: object()))
    calls=[]
    async def run(config):
        calls.append('committed')
        if business_error:
            raise LookupError('business exception')
        return {'status': 'answered'}
    if business_error:
        with pytest.raises(LookupError):
            asyncio.run(obs.trace_turn('a', {}, {}, run))
    else:
        assert asyncio.run(obs.trace_turn('a', {}, {}, run))['status'] == 'answered'
    assert calls == ['committed']
