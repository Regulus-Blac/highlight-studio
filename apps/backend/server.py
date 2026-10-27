"""Local D/E integration server; deploy behind E's authentication boundary."""

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from services.model_adapter.adapter import MockAdapter, validate_request
from services.model_adapter.command import create_adapter
from services.task_service.store import TaskError, TaskStore
from services.task_service.worker import TaskWorker
from services.media.probe import MediaError, probe
from services.media.service import MediaService
from services.media.http import send_file

ROOT = Path(__file__).resolve().parents[2]


class ProductServer(ThreadingHTTPServer):
    worker = None

    def server_close(self):
        if self.worker:
            self.worker.close()
        super().server_close()


def make_server(host="127.0.0.1", port=8000, *, runtime=None, adapter=None, start_worker=True, max_upload_bytes=512 * 1024 * 1024, cors_origin=None, allow_mock_media=False):
    store = TaskStore(runtime or ROOT / "runtime")
    adapter = adapter or MockAdapter()
    legacy_jobs = {}
    media = MediaService(store, allow_mock=allow_mock_media)

    class Handler(BaseHTTPRequestHandler):
        def cors(self):
            if cors_origin and self.headers.get("Origin") == cors_origin:
                self.send_header("Access-Control-Allow-Origin", cors_origin)
                self.send_header("Vary", "Origin")

        def respond(self, status, body):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, code, message):
            self.respond(status, {"error": {"code": code, "message": message}})

        def content_length(self, maximum):
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise TaskError(400, "invalid_request", "Invalid Content-Length") from error
            if size <= 0:
                raise TaskError(400, "invalid_request", "A non-empty request body is required")
            if size > maximum:
                raise TaskError(413, "request_too_large", "Request exceeds the configured size limit")
            return size

        def read_json(self):
            size = self.content_length(65536)
            try:
                return json.loads(self.rfile.read(size))
            except (ValueError, UnicodeDecodeError) as error:
                raise TaskError(400, "invalid_request", "Request must contain valid JSON") from error

        def do_OPTIONS(self):
            if not cors_origin or self.headers.get("Origin") != cors_origin:
                self.error(403, "origin_not_allowed", "Origin is not configured for local integration")
                return
            self.send_response(204)
            self.cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Idempotency-Key")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    data = (ROOT / "apps/frontend/index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif path == "/health":
                    self.respond(200, {"status": "ok", "engine": adapter.engine})
                elif path == "/ready":
                    ready = bool(server.worker and server.worker.thread and server.worker.thread.is_alive() and adapter.available)
                    self.respond(200 if ready else 503, {"ready": ready, "engine": adapter.engine, "mock": adapter.mock})
                elif path == "/api/v1/model":
                    self.respond(200, {"api_version": "1", "engine": adapter.engine, "mock": adapter.mock, "available": adapter.available, "prediction_api_version": "1"})
                elif path.startswith("/api/v1/uploads/") and path.endswith("/metadata"):
                    parts = path.removeprefix("/api/v1/uploads/").split("/")
                    if len(parts) != 2:
                        raise TaskError(404, "not_found", "Unknown route")
                    _, source = media.upload_source(parts[0])
                    self.respond(200, probe(source))
                elif path.startswith("/api/v1/tasks/"):
                    parts = path.removeprefix("/api/v1/tasks/").split("/")
                    if len(parts) == 1:
                        self.respond(200, store.get_task(parts[0]))
                    elif len(parts) == 2 and parts[1] == "result":
                        task = store.get_task(parts[0])
                        if task["status"] != "succeeded":
                            raise TaskError(409, "result_not_available", "Prediction is available only after the task succeeds")
                        self.respond(200, task["result"])
                    elif len(parts) == 2 and parts[1] == "preview":
                        self.respond(200, media.preview(parts[0]))
                    elif len(parts) == 2 and parts[1] == "source":
                        _, upload, source = media.source(parts[0])
                        send_file(self, source, "video/" + ("quicktime" if source.suffix == ".mov" else "mp4" if source.suffix == ".mp4" else "webm" if source.suffix == ".webm" else "x-matroska" if source.suffix == ".mkv" else "x-msvideo"), upload["sha256"])
                    else:
                        raise TaskError(404, "not_found", "Unknown route")
                elif path.startswith("/api/v1/exports/"):
                    parts = path.removeprefix("/api/v1/exports/").split("/")
                    if len(parts) == 1:
                        self.respond(200, media.get(parts[0]))
                    elif len(parts) == 2 and parts[1] == "file":
                        artifact, item = media.artifact_path(parts[0])
                        send_file(self, artifact, "video/mp4" if item["format"] == "mp4" else "application/x-ndjson", item["artifact"]["sha256"])
                    else:
                        raise TaskError(404, "not_found", "Unknown route")
                elif path.startswith("/api/v1/jobs/"):
                    job = legacy_jobs.get(path.removeprefix("/api/v1/jobs/"))
                    if job is None:
                        raise TaskError(404, "not_found", "Unknown legacy mock job")
                    self.respond(200, job)
                else:
                    raise TaskError(404, "not_found", "Unknown route")
            except TaskError as error:
                self.error(error.status, error.code, str(error))
            except MediaError as error:
                self.error(422, error.code, str(error))

        def upload(self):
            size = self.content_length(max_upload_bytes)
            names = parse_qs(urlsplit(self.path).query).get("filename", [])
            if len(names) != 1:
                raise TaskError(400, "invalid_request", "Provide one filename query parameter")
            filename = names[0].replace("\\", "/").rsplit("/", 1)[-1]
            if not filename or len(filename) > 255 or any(ord(c) < 32 for c in filename):
                raise TaskError(400, "invalid_request", "Invalid filename")
            suffix = Path(filename).suffix.lower()
            if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                raise TaskError(415, "unsupported_video_type", "Supported filename extensions: mp4, mov, mkv, webm, avi")
            upload_id = uuid4().hex
            target = store.uploads / (upload_id + suffix)
            partial = target.with_suffix(target.suffix + ".partial")
            digest, remaining = hashlib.sha256(), size
            try:
                self.connection.settimeout(30)
                with partial.open("xb") as destination:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise TaskError(400, "incomplete_upload", "Upload ended before Content-Length bytes were received")
                        destination.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                partial.replace(target)
                response = store.add_upload(upload_id, filename, target, size, digest.hexdigest())
            except (OSError, TaskError):
                partial.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                raise
            self.respond(201, response)

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                if path == "/api/v1/uploads":
                    self.upload()
                elif path == "/api/v1/tasks":
                    task, created = store.create_task(self.read_json(), adapter, self.headers.get("Idempotency-Key"))
                    self.respond(202 if created else 200, task)
                elif path == "/api/v1/exports":
                    item, created = media.create(self.read_json(), self.headers.get("Idempotency-Key"))
                    self.respond(202 if created else 200, item)
                elif path.startswith("/api/v1/tasks/") and path.endswith("/preview"):
                    parts = path.removeprefix("/api/v1/tasks/").split("/")
                    payload = self.read_json()
                    if len(parts) != 2 or not isinstance(payload, dict) or set(payload) != {"api_version", "segments"} or payload["api_version"] != "1":
                        raise TaskError(400, "invalid_request", "Edited preview requires api_version=1 and segments")
                    self.respond(200, media.preview(parts[0], payload["segments"]))
                elif path.startswith("/api/v1/tasks/") and path.endswith("/cancel"):
                    parts = path.removeprefix("/api/v1/tasks/").split("/")
                    if len(parts) != 2:
                        raise TaskError(404, "not_found", "Unknown route")
                    self.respond(200, store.cancel(parts[0]))
                elif path == "/api/v1/jobs":
                    if not adapter.mock:
                        raise TaskError(410, "legacy_mock_only", "Use uploads and /api/v1/tasks for configured inference")
                    request = self.read_json()
                    try:
                        validate_request(request)
                    except ValueError as error:
                        raise TaskError(400, "invalid_request", str(error)) from error
                    job_id = uuid4().hex
                    job = {"job_id": job_id, "status": "succeeded", "result": MockAdapter().predict(request)}
                    legacy_jobs[job_id] = job
                    self.respond(201, job)
                else:
                    raise TaskError(404, "not_found", "Unknown route")
            except TaskError as error:
                self.error(error.status, error.code, str(error))
            except MediaError as error:
                self.error(422, error.code, str(error))
            except (socket.timeout, TimeoutError):
                self.error(408, "request_timeout", "Upload timed out")
            except Exception:
                self.error(500, "internal_error", "Task service could not complete this request")

    server = ProductServer((host, port), Handler)
    server.store = store
    server.adapter = adapter
    server.media = media
    if start_worker:
        server.worker = TaskWorker(store, adapter, media)
        try:
            server.worker.start()
        except Exception:
            server.server_close()
            raise
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime")
    parser.add_argument("--adapter", choices=("mock", "mock-media", "command"), default="mock")
    parser.add_argument("--adapter-config", type=Path)
    parser.add_argument("--cors-origin", help="Exact origin of D's local development frontend")
    parser.add_argument("--max-upload-mb", type=int, default=512)
    parser.add_argument("--allow-mock-media", action="store_true", help="Development only: allow explicitly labelled synthetic exports")
    args = parser.parse_args()
    if args.max_upload_mb <= 0:
        parser.error("--max-upload-mb must be positive")
    adapter = create_adapter(args.adapter, args.adapter_config, args.runtime / "inference")
    with make_server(args.host, args.port, runtime=args.runtime, adapter=adapter, cors_origin=args.cors_origin, max_upload_bytes=args.max_upload_mb * 1024 * 1024, allow_mock_media=args.allow_mock_media) as server:
        print(f"Integration server: http://{args.host}:{server.server_port} engine={adapter.engine} mock={adapter.mock}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
