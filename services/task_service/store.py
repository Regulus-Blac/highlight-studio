"""SQLite-backed uploads and durable single-worker prediction tasks."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from services.model_adapter.adapter import validate_request


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TaskError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code


class TaskStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.uploads = self.root / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "tasks.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS uploads (
                    upload_id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, request TEXT NOT NULL,
                    adapter_identity TEXT NOT NULL, engine TEXT NOT NULL, mock INTEGER NOT NULL,
                    status TEXT NOT NULL, result TEXT, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE
                );
                CREATE INDEX IF NOT EXISTS tasks_queue ON tasks(status, created_at);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def add_upload(self, upload_id, filename, path, size, digest):
        timestamp = now()
        with self.connect() as db:
            db.execute("INSERT INTO uploads VALUES (?,?,?,?,?,?)", (upload_id, filename, str(path), size, digest, timestamp))
        return {"api_version": "1", "upload_id": upload_id, "filename": filename, "size_bytes": size, "sha256": digest, "created_at": timestamp}

    def create_task(self, payload, adapter, idempotency_key=None):
        if not isinstance(payload, dict) or set(payload) != {"api_version", "upload_id", "target_ratio"}:
            raise TaskError(400, "invalid_request", "Expected api_version, upload_id and target_ratio")
        if not isinstance(payload["upload_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", payload["upload_id"]):
            raise TaskError(400, "invalid_request", "upload_id must be the identifier returned by the upload API")
        if idempotency_key is not None and (not idempotency_key.strip() or len(idempotency_key) > 128):
            raise TaskError(400, "invalid_request", "Idempotency-Key must contain 1 to 128 characters")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            upload = db.execute("SELECT * FROM uploads WHERE upload_id=?", (payload["upload_id"],)).fetchone()
            if upload is None:
                raise TaskError(404, "upload_not_found", "Unknown upload")
            request = {"api_version": payload["api_version"], "video_id": upload["upload_id"], "video_path": upload["path"], "target_ratio": payload["target_ratio"]}
            try:
                validate_request(request)
            except ValueError as error:
                raise TaskError(400, "invalid_request", str(error)) from error
            encoded = json.dumps(request, sort_keys=True)
            if idempotency_key:
                old = db.execute("SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)).fetchone()
                if old:
                    if old["request"] != encoded:
                        raise TaskError(409, "idempotency_conflict", "Idempotency-Key was already used for a different request")
                    return self.public(old), False
            task_id, timestamp = uuid4().hex, now()
            db.execute("INSERT INTO tasks (task_id,request,adapter_identity,engine,mock,status,created_at,updated_at,idempotency_key) VALUES (?,?,?,?,?,'queued',?,?,?)", (task_id, encoded, adapter.identity, adapter.engine, int(adapter.mock), timestamp, timestamp, idempotency_key))
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return self.public(row), True

    @staticmethod
    def public(row):
        request, status = json.loads(row["request"]), row["status"]
        stages = {"queued": "queued", "running": "predicting", "succeeded": "completed", "failed": "failed", "cancelled": "cancelled"}
        return {
            "api_version": "1", "task_id": row["task_id"], "video_id": request["video_id"],
            "target_ratio": request["target_ratio"], "status": status,
            "progress": {"stage": stages[status], "percent": 100 if status == "succeeded" else 0 if status == "queued" else None},
            "engine": row["engine"], "mock": bool(row["mock"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": json.loads(row["error"]) if row["error"] else None,
        }

    def get_task(self, task_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise TaskError(404, "task_not_found", "Unknown task")
        return self.public(row)

    def cancel(self, task_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise TaskError(404, "task_not_found", "Unknown task")
            if row["status"] not in {"queued", "cancelled"}:
                raise TaskError(409, "task_not_cancellable", "Only queued tasks can be cancelled")
            db.execute("UPDATE tasks SET status='cancelled',updated_at=? WHERE task_id=?", (now(), task_id))
        return self.get_task(task_id)

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY created_at,task_id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE tasks SET status='running',updated_at=? WHERE task_id=?", (now(), row["task_id"]))
        return dict(row)

    def finish(self, task_id, result=None, error=None):
        with self.connect() as db:
            db.execute("UPDATE tasks SET status=?,result=?,error=?,updated_at=? WHERE task_id=? AND status='running'", ("failed" if error else "succeeded", json.dumps(result) if result is not None else None, json.dumps(error) if error else None, now(), task_id))

    def recover_interrupted(self):
        error = json.dumps({"code": "worker_interrupted", "message": "Previous worker stopped before completing this task; create a new task to retry", "retryable": True})
        with self.connect() as db:
            db.execute("UPDATE tasks SET status='failed',error=?,updated_at=? WHERE status='running'", (error, now()))
