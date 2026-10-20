"""Authenticated retail tool dispatcher with durable user confirmation."""

import hashlib
import json
import time
import uuid
from pathlib import Path

from retailflow.storage import Storage
from retailflow.tools import IDENTITY_TOOLS, SCHEMAS, WRITE_TOOLS, invoke, valid_arguments
from retailflow.upstream.data_model import RetailDB
from retailflow.upstream.paths import RETAIL_DB_PATH


PROPOSAL_TTL_SECONDS = 900


class PreExecutionTemporaryFailure(Exception):
    """The tool did not start, so a read may be retried with identical arguments."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _result(code, message):
    return {"ok": False, "code": code, "message": message}


def _business_error(exc):
    message = str(exc)
    return _result("NOT_FOUND" if "not found" in message.lower() else "POLICY_DENIED", message)


class RetailEnvironment:
    def __init__(self, store_path: Path = Path("runtime/business.sqlite")):
        self.store_path = Path(store_path)
        self.storage = Storage(self.store_path)

    def create_session(self, thread_id: str, initial_state_ref: str) -> None:
        source = RETAIL_DB_PATH.read_bytes()
        expected = f"retail-db@{hashlib.sha256(source).hexdigest()}"
        if initial_state_ref != expected:
            raise ValueError("initial state reference does not match fixed retail DB")
        with self.storage.transaction() as db:
            db.execute("INSERT OR IGNORE INTO sessions(thread, business_json) VALUES (?, ?)", (thread_id, source.decode()))

    def _session(self, db, thread_id):
        row = db.execute("SELECT * FROM sessions WHERE thread = ?", (thread_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return row

    def get_order(self, thread_id: str, order_id: str) -> dict:
        with self.storage.transaction() as db:
            business = json.loads(self._session(db, thread_id)["business_json"])
        return business["orders"][order_id]

    def state_hash(self, thread_id: str) -> str:
        with self.storage.transaction() as db:
            business = json.loads(self._session(db, thread_id)["business_json"])
        return hashlib.sha256(_json(business).encode()).hexdigest()

    def load_session(self, thread_id: str) -> dict:
        with self.storage.transaction() as db:
            row = db.execute("SELECT * FROM sessions WHERE thread = ?", (thread_id,)).fetchone()
            receipt = db.execute("SELECT name, result_json FROM receipts WHERE thread = ? ORDER BY rowid DESC LIMIT 1",
                                 (thread_id,)).fetchone()
        if row is None:
            return {}
        business = json.loads(row["business_json"])
        target = business["orders"].get(row["target_order_id"], {})
        return {"business": business, "user_id": row["user_id"],
                "target_order_id": row["target_order_id"],
                "target_order_status": target.get("status"),
                "pending_action": self.pending_proposal(thread_id),
                "last_receipt": {"name": receipt["name"], "result": json.loads(receipt["result_json"])} if receipt else None}

    def receipt_for_proposal(self, thread_id: str, proposal_id: str):
        with self.storage.transaction() as db:
            row = db.execute("SELECT result_json FROM receipts WHERE thread = ? AND "
                             "json_extract(result_json, '$.proposal_id') = ?", (thread_id, proposal_id)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def pending_proposal(self, thread_id: str):
        with self.storage.transaction() as db:
            pending = self._session(db, thread_id)["pending_json"]
            proposal = json.loads(pending) if pending else None
            if proposal and time.time() - proposal["created_at"] > PROPOSAL_TTL_SECONDS:
                db.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                return None
            return proposal

    def invalidate_pending(self, thread_id: str) -> None:
        with self.storage.transaction() as db:
            row = self._session(db, thread_id)
            pending = json.loads(row["pending_json"]) if row["pending_json"] else None
            if pending:
                db.execute("UPDATE sessions SET pending_json = NULL, target_order_id = NULL WHERE thread = ?", (thread_id,))

    def confirm_action(self, thread_id: str, proposal_id: str, accepted: bool) -> dict:
        if type(accepted) is not bool:
            return _result("INVALID_CONFIRMATION", "Confirmation must be an explicit yes or no.")
        with self.storage.transaction() as db:
            row = self._session(db, thread_id)
            pending = json.loads(row["pending_json"]) if row["pending_json"] else None
            if not pending or pending["proposal_id"] != proposal_id:
                return _result("INVALID_CONFIRMATION", "Proposal does not match this session.")
            if time.time() - pending["created_at"] > PROPOSAL_TTL_SECONDS:
                db.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                return _result("INVALID_CONFIRMATION", "Proposal expired; request a new confirmation.")
            if not accepted:
                db.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                return {"ok": True, "code": "DECLINED"}
            pending["confirmed"] = True
            db.execute("UPDATE sessions SET pending_json = ? WHERE thread = ?", (_json(pending), thread_id))
            return {"ok": True, "code": "CONFIRMED"}

    def dispatch_tool(self, thread_id: str, name: str, args: dict) -> dict:
        if name not in SCHEMAS:
            return _result("UNKNOWN_TOOL", "This tool is outside the fixed retail scope.")
        if not valid_arguments(name, args):
            return _result("INVALID_ARGUMENT", "Arguments do not match the fixed tool schema.")
        arguments = _json(args)
        with self.storage.transaction() as sql:
            row = self._session(sql, thread_id)
            business = json.loads(row["business_json"])
            user_id = row["user_id"]
            pending = json.loads(row["pending_json"]) if row["pending_json"] else None
            if pending and time.time() - pending["created_at"] > PROPOSAL_TTL_SECONDS:
                sql.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                pending = None
            changed_target = "order_id" in args and pending and args["order_id"] != pending["args"].get("order_id")
            changed_profile = name == "get_user_details" and args["user_id"] != user_id
            changed_write = name in WRITE_TOOLS and pending and (pending["name"] != name or pending["args"] != args)
            if pending and (changed_target or changed_profile or changed_write):
                sql.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                pending = None
            if name in WRITE_TOOLS:
                receipt = sql.execute("SELECT result_json FROM receipts WHERE thread = ? AND name = ? AND args_json = ?", (thread_id, name, arguments)).fetchone()
                if receipt:
                    return json.loads(receipt["result_json"])
            if name not in IDENTITY_TOOLS and not user_id:
                return _result("AUTH_REQUIRED", "Locate the user by email or name and ZIP code first.")
            if name == "get_user_details" and args["user_id"] != user_id:
                return _result("POLICY_DENIED", "Only the authenticated user's profile can be read.")
            if "order_id" in args:
                order = business["orders"].get(args["order_id"])
                if order and order["user_id"] != user_id:
                    return _result("POLICY_DENIED", "This order belongs to another user.")
            snapshot = RetailDB.model_validate(business)
            if name in WRITE_TOOLS:
                if not pending or not pending["confirmed"]:
                    try:
                        invoke(snapshot, name, args)  # Validate on a disposable copy, before proposing.
                    except ValueError as exc:
                        return _business_error(exc)
                    except Exception as exc:
                        return _result("TOOL_ERROR", f"{type(exc).__name__}: {exc}")
                    if not pending:
                        pending = {"proposal_id": uuid.uuid4().hex, "name": name, "args": args, "confirmed": False, "created_at": time.time()}
                        sql.execute("UPDATE sessions SET pending_json = ? WHERE thread = ?", (_json(pending), thread_id))
                    return {"ok": False, "code": "CONFIRMATION_REQUIRED", "proposal_id": pending["proposal_id"], "proposal": pending}
                try:
                    completed = RetailDB.model_validate(business)
                    result = invoke(completed, name, args)
                except ValueError as exc:
                    return _business_error(exc)
                except Exception as exc:
                    return _result("TOOL_ERROR", f"{type(exc).__name__}: {exc}")
                response = {"ok": True, "code": "OK", "data": result, "proposal_id": pending["proposal_id"]}
                key = hashlib.sha256(f"{thread_id}:{pending['proposal_id']}:{name}:{arguments}".encode()).hexdigest()
                sql.execute("INSERT INTO receipts(action_key, thread, name, args_json, result_json) VALUES (?, ?, ?, ?, ?)", (key, thread_id, name, arguments, _json(response)))
                sql.execute("UPDATE sessions SET business_json = ?, pending_json = NULL WHERE thread = ?", (completed.model_dump_json(), thread_id))
                return response
            try:
                result = invoke(snapshot, name, args)
            except PreExecutionTemporaryFailure as exc:
                return _result("TEMPORARY_FAILURE", str(exc))
            except ValueError as exc:
                return _result("NOT_FOUND", str(exc))
            except Exception as exc:
                return _result("TOOL_ERROR", f"{type(exc).__name__}: {exc}")
            if name in IDENTITY_TOOLS:
                if user_id and user_id != result:
                    if pending:
                        sql.execute("UPDATE sessions SET pending_json = NULL WHERE thread = ?", (thread_id,))
                    return _result("POLICY_DENIED", "This session is bound to one user.")
                if not user_id:
                    sql.execute("UPDATE sessions SET user_id = ? WHERE thread = ?", (result, thread_id))
            if name == "get_order_details":
                sql.execute("UPDATE sessions SET target_order_id = ? WHERE thread = ?", (args["order_id"], thread_id))
            return {"ok": True, "code": "OK", "data": result}


def create_session(thread_id: str, initial_state_ref: str) -> None:
    RetailEnvironment().create_session(thread_id, initial_state_ref)


def dispatch_tool(thread_id: str, name: str, args: dict) -> dict:
    return RetailEnvironment().dispatch_tool(thread_id, name, args)


def confirm_action(thread_id: str, proposal_id: str, accepted: bool) -> dict:
    return RetailEnvironment().confirm_action(thread_id, proposal_id, accepted)
