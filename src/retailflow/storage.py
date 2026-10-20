"""SQLite transaction boundary for a session's business snapshot and receipts."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Storage:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (thread TEXT PRIMARY KEY, business_json TEXT NOT NULL, user_id TEXT, pending_json TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS receipts (action_key TEXT PRIMARY KEY, thread TEXT NOT NULL, name TEXT NOT NULL, args_json TEXT NOT NULL, result_json TEXT NOT NULL)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS receipt_lookup ON receipts(thread, name, args_json)")
            db.execute("CREATE TABLE IF NOT EXISTS conversations (thread TEXT PRIMARY KEY, messages_json TEXT NOT NULL)")

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
