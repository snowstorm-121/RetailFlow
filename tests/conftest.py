import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / "vendor/tau_retail/source/data/tau2/domains/retail/db.json"
STATE_REF = f"retail-db@{json.loads((ROOT / 'data/manifests/dataset.json').read_text())['source_db_sha256']}"


@pytest.fixture
def env(tmp_path):
    from retailflow.environment import RetailEnvironment

    worker = RetailEnvironment(tmp_path / "business.sqlite")
    worker.create_session("session-a", STATE_REF)
    result = worker.dispatch_tool("session-a", "find_user_id_by_name_zip", {
        "first_name": "Yara", "last_name": "Muller", "zip": "85041",
    })
    assert result["ok"] and result["data"] == "yara_muller_8652"
    return worker
