"""Preflight only: reject drift before a frozen evaluation starts."""

import hashlib
import json
from pathlib import Path

from retailflow.dataset import ROOT
from retailflow.evaluation import build_manifest


def verify_frozen(config_path: Path, manifest_path: Path) -> dict:
    frozen = json.loads(manifest_path.read_text())
    checksum = frozen.pop('manifest_sha256')
    actual = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False).encode()).hexdigest()
    if actual != checksum:
        raise ValueError('Freeze checksum mismatch')
    parent = frozen['parent_manifest']
    if hashlib.sha256((ROOT / parent['path']).read_bytes()).hexdigest() != parent['sha256']:
        raise ValueError('Parent manifest mismatch')
    current = build_manifest(config_path, json.loads(config_path.read_text()))
    if current != frozen['environment']:
        changed = sorted(key for key in set(current) | set(frozen['environment'])
                         if current.get(key) != frozen['environment'].get(key))
        raise ValueError(f'Frozen environment mismatch: {changed}')
    return {**frozen, 'manifest_sha256': checksum}
