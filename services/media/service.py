"""Persistent reference export queue; E can reuse the core behind business v2."""

import json
import logging
from pathlib import Path
import re
import threading
from uuid import uuid4

from services.media.plan import make_plan
from services.media.probe import MediaError, digest_file, probe
from services.media.render import render_jsonl, render_mp4
from services.task_service.store import TaskError, now


class MediaService:
    def __init__(self, store, *, allow_mock=False):
        self.store, self.allow_mock = store, allow_mock
        self.stopping = threading.Event()
        self.root = store.root / "exports"
        self.root.mkdir(exist_ok=True)
        with store.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS exports (
                export_id TEXT PRIMARY KEY, request TEXT NOT NULL, snapshot TEXT NOT NULL,
                status TEXT NOT NULL, artifact TEXT, error TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, idempotency_key TEXT UNIQUE)""")

    def source(self, task_id):
        task = self.store.get_task(task_id)
        if task["status"] != "succeeded":
            raise TaskError(409, "result_not_available", "Prediction must succeed before preview/export")
        row, path = self.upload_source(task["video_id"])
        return task, row, path

    def upload_source(self, upload_id):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM uploads WHERE upload_id=?", (upload_id,)).fetchone()
        if row is None:
            raise TaskError(404, "upload_not_found", "Source upload is unavailable")
        path = Path(row["path"])
        if not path.is_file() or digest_file(path) != row["sha256"]:
            raise MediaError("source_changed", "Source bytes no longer match the uploaded identity")
        return row, path

    def preview(self, task_id, edits=None):
        task, upload, source = self.source(task_id)
        plan = make_plan(task["result"], probe(source), edits)
        return {**plan, "source_sha256": upload["sha256"], "source_url": f"/api/v1/tasks/{task_id}/source"}

    @staticmethod
    def public(row):
        snapshot = json.loads(row["snapshot"])
        artifact = json.loads(row["artifact"]) if row["artifact"] else None
        return {"api_version": "1", "export_id": row["export_id"], "task_id": json.loads(row["request"])["task_id"],
                "plan_id": snapshot["plan"]["plan_id"], "engine": snapshot["plan"]["engine"], "mock": snapshot["plan"]["mock"],
                "format": json.loads(row["request"])["format"], "status": row["status"], "artifact": artifact,
                "error": json.loads(row["error"]) if row["error"] else None, "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def get(self, export_id):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM exports WHERE export_id=?", (export_id,)).fetchone()
        if row is None:
            raise TaskError(404, "export_not_found", "Unknown export")
        return self.public(row)

    def create(self, request, key=None):
        if not isinstance(request, dict) or not {"api_version", "task_id", "format"} <= set(request) or set(request) - {"api_version", "task_id", "format", "segments"} or request["api_version"] != "1" or not isinstance(request["task_id"], str) or not re.fullmatch(r"[0-9a-f]{32}", request["task_id"]) or request["format"] not in ("mp4", "jsonl"):
            raise TaskError(400, "invalid_request", "Expected api_version=1, task_id, format=mp4|jsonl and optional segments")
        if "segments" in request and not isinstance(request["segments"], list):
            raise MediaError("invalid_segments", "Explicit segments must be an array; omit the field to use suggestions")
        if key is not None and (not key.strip() or len(key) > 128):
            raise TaskError(400, "invalid_request", "Idempotency-Key must contain 1 to 128 characters")
        encoded = json.dumps(request, sort_keys=True)
        # Replaying an already-created export does not re-probe changed sources.
        with self.store.connect() as db:
            old = db.execute("SELECT * FROM exports WHERE idempotency_key=?", (key,)).fetchone() if key else None
        if old:
            if old["request"] != encoded:
                raise TaskError(409, "idempotency_conflict", "Key belongs to a different export request")
            return self.public(old), False
        task, upload, source = self.source(request["task_id"])
        if task["mock"] and not self.allow_mock:
            raise TaskError(409, "mock_export_disabled", "Mock exports require an explicit development-only server option")
        plan = make_plan(task["result"], probe(source), request.get("segments"))
        if request["format"] == "mp4" and not plan["frame_crops"]:
            raise MediaError("no_enabled_segments", "No enabled frames to export")
        snapshot = json.dumps({"source": str(source), "source_sha256": upload["sha256"], "result": task["result"], "plan": plan}, sort_keys=True)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM exports WHERE idempotency_key=?", (key,)).fetchone() if key else None
            if old:
                if old["request"] != encoded:
                    raise TaskError(409, "idempotency_conflict", "Key belongs to a different export request")
                return self.public(old), False
            # Serial reference worker; one active export for each prediction.
            active = db.execute("SELECT request FROM exports WHERE status IN ('queued','running')").fetchall()
            if any(json.loads(r["request"])["task_id"] == request["task_id"] for r in active):
                raise TaskError(409, "export_active", "This prediction already has an active export")
            export_id, timestamp = uuid4().hex, now()
            db.execute("INSERT INTO exports VALUES (?,?,?,'queued',NULL,NULL,?,?,?)", (export_id, encoded, snapshot, timestamp, timestamp, key))
            row = db.execute("SELECT * FROM exports WHERE export_id=?", (export_id,)).fetchone()
        return self.public(row), True

    def recover(self):
        with self.store.connect() as db:
            db.execute("UPDATE exports SET status='failed',error=?,updated_at=? WHERE status='running'", (json.dumps({"code": "worker_interrupted", "message": "Export worker stopped; retry with a new key", "retryable": True}), now()))

    def run_once(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM exports WHERE status='queued' ORDER BY created_at,export_id LIMIT 1").fetchone()
            if row is None:
                return False
            db.execute("UPDATE exports SET status='running',updated_at=? WHERE export_id=?", (now(), row["export_id"]))
        artifact, error = None, None
        directory = self.root / row["export_id"]
        request, snapshot = json.loads(row["request"]), json.loads(row["snapshot"])
        partial = directory / ("partial." + request["format"])
        try:
            directory.mkdir(exist_ok=True)
            if digest_file(snapshot["source"]) != snapshot["source_sha256"]:
                raise MediaError("source_changed", "Source bytes changed after the export snapshot")
            if request["format"] == "mp4":
                render_mp4(snapshot["source"], snapshot["plan"], partial, cancel_event=self.stopping)
            else:
                render_jsonl(snapshot["result"], snapshot["plan"], partial)
            if digest_file(snapshot["source"]) != snapshot["source_sha256"]:
                raise MediaError("source_changed", "Source bytes changed during export")
            final = directory / ("artifact." + request["format"])
            partial.replace(final)
            artifact = {"url": f"/api/v1/exports/{row['export_id']}/file", "size_bytes": final.stat().st_size, "sha256": digest_file(final),
                        "output": snapshot["plan"]["output"], "duration_seconds": snapshot["plan"]["duration_seconds"]}
        except MediaError as failure:
            error = {"code": failure.code, "message": str(failure), "retryable": False}
        except Exception:
            logging.exception("Export %s failed", row["export_id"])
            error = {"code": "render_failed", "message": "Export failed; inspect private worker logs", "retryable": False}
        finally:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                logging.warning("Could not clean temporary output for export %s", row["export_id"], exc_info=True)
        with self.store.connect() as db:
            db.execute("UPDATE exports SET status=?,artifact=?,error=?,updated_at=? WHERE export_id=? AND status='running'", ("failed" if error else "succeeded", json.dumps(artifact) if artifact else None, json.dumps(error) if error else None, now(), row["export_id"]))
        return True

    def close(self):
        self.stopping.set()

    def artifact_path(self, export_id):
        item = self.get(export_id)
        if item["status"] != "succeeded":
            raise TaskError(409, "artifact_not_available", "Export has not succeeded")
        return self.root / item["export_id"] / ("artifact." + item["format"]), item
