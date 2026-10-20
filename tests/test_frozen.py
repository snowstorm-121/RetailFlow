import copy
import hashlib
import json
import subprocess
import sys

import pytest



def freeze_file(tmp_path, environment):
    parent = tmp_path / 'parent.json'
    parent.write_text('{}')
    frozen = {'environment': environment, 'parent_manifest': {'path': str(parent),
              'sha256': hashlib.sha256(parent.read_bytes()).hexdigest()}}
    frozen['manifest_sha256'] = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':'),
                                                        ensure_ascii=False).encode()).hexdigest()
    path = tmp_path / 'freeze.json'
    path.write_text(json.dumps(frozen))
    return path


@pytest.mark.parametrize('changed', [None, 'config', 'file_hashes', 'ollama_runtime', 'model', 'variants'])
def test_frozen_gate_rejects_environment_drift_before_running(tmp_path, monkeypatch, changed):
    from retailflow import frozen

    original = {'config': {'temperature': 0}, 'file_hashes': {'scorer.py': 'abc'},
                'ollama_runtime': {'version': 'fixed'}, 'model': {'digest': 'fixed'},
                'variants': {'B0': {'max_retry_per_call': 0}}}
    path = freeze_file(tmp_path, original)
    current = copy.deepcopy(original)
    if changed:
        current[changed] = {'changed': True}
    monkeypatch.setattr(frozen, 'build_manifest', lambda *_: current)
    config = tmp_path / 'config.json'
    config.write_text('{}')
    if changed:
        with pytest.raises(ValueError, match='Frozen environment mismatch'):
            frozen.verify_frozen(config, path)
    else:
        assert frozen.verify_frozen(config, path)['environment'] == original


def test_frozen_gate_rejects_tampered_manifest_and_parent(tmp_path, monkeypatch):
    from retailflow import frozen

    path = freeze_file(tmp_path, {})
    monkeypatch.setattr(frozen, 'build_manifest', lambda *_: {})
    config = tmp_path / 'config.json'
    config.write_text('{}')
    (tmp_path / 'parent.json').write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Parent manifest mismatch'):
        frozen.verify_frozen(config, path)
    path = freeze_file(tmp_path, {})
    payload = json.loads(path.read_text());payload['environment'] = {'changed': True}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Freeze checksum mismatch'):
        frozen.verify_frozen(config, path)


@pytest.mark.parametrize('extra', [[], ['--frozen', '--limit', '1']])
def test_test_cli_refuses_unfrozen_or_partial_run_without_contacting_model(extra):
    result = subprocess.run([sys.executable, '-m', 'retailflow.cli', 'evaluate', '--split', 'test',
                             '--variant', 'B0', *extra], capture_output=True, text=True)
    assert result.returncode == 2
    assert ('requires --frozen' if not extra else 'cannot use --limit') in result.stderr
