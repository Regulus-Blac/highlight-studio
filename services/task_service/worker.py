"""One worker per runtime directory, with process locking and crash recovery."""

import json
import os
import threading

from services.model_adapter.adapter import AdapterError, validate_result
from services.media.probe import MediaError, digest_file, probe


class TaskWorker:
    def __init__(self, store, adapter, media=None):
        self.store, self.adapter, self.media = store, adapter, media
        self.stopping = threading.Event()
        self.thread = None
        self.lock_file = None

    def start(self):
        self.lock_file = (self.store.root / "worker.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.lock_file.seek(0)
                self.lock_file.write(b"0")
                self.lock_file.flush()
                self.lock_file.seek(0)
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.lock_file.close()
            raise RuntimeError("A worker already owns this runtime directory") from error
        self.store.recover_interrupted()
        if self.media:
            self.media.recover()
        self.thread = threading.Thread(target=self.run, name="prediction-worker", daemon=True)
        self.thread.start()

    def run_once(self):
        row = self.store.claim()
        if row is None:
            return False
        try:
            if row["adapter_identity"] != self.adapter.identity:
                raise AdapterError("model_version_unavailable", "The task's pinned model version is unavailable; create a new task for the current version")
            if not self.adapter.available:
                raise AdapterError("model_unavailable", "The configured real model is unavailable")
            request = json.loads(row["request"])
            try:
                metadata = None
                if not self.adapter.mock:
                    with self.store.connect() as db:
                        upload = db.execute("SELECT sha256 FROM uploads WHERE upload_id=?", (request["video_id"],)).fetchone()
                    if upload is None or digest_file(request["video_path"]) != upload["sha256"]:
                        raise MediaError("source_changed", "Source no longer matches the uploaded bytes")
                    metadata = probe(request["video_path"])
                result = self.adapter.predict(request)
                validate_result(result, request)
                if result["engine"] != row["engine"] or result["mock"] != bool(row["mock"]):
                    raise ValueError("Adapter changed the pinned model identity")
                if not result["mock"]:
                    from services.media.plan import make_plan
                    make_plan(result, metadata)
                    if digest_file(request["video_path"]) != upload["sha256"]:
                        raise MediaError("source_changed", "Source changed during inference")
            except ValueError as error:
                raise AdapterError("invalid_prediction", "Model adapter returned an invalid v1 prediction") from error
            self.store.finish(row["task_id"], result=result)
        except AdapterError as error:
            self.store.finish(row["task_id"], error=error.as_dict())
        except MediaError as error:
            self.store.finish(row["task_id"], error={"code": error.code, "message": str(error), "retryable": False})
        except Exception:
            import logging
            logging.exception("Prediction %s failed", row["task_id"])
            self.store.finish(row["task_id"], error={"code": "inference_failed", "message": "Model adapter failed", "retryable": False})
        return True

    def run(self):
        try:
            while not self.stopping.is_set():
                predicted = self.run_once()
                exported = self.media.run_once() if self.media and not self.stopping.is_set() else False
                if not predicted and not exported:
                    self.stopping.wait(0.1)
        finally:
            self.lock_file.close()

    def close(self):
        self.stopping.set()
        if self.media:
            self.media.close()
        close_adapter = getattr(self.adapter, "close", None)
        if close_adapter:
            close_adapter()
        if self.thread:
            self.thread.join(timeout=5)
