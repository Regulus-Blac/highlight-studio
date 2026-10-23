"""仅用于本机产品联调的 HTTP 骨架，运行 python -m apps.backend.server。"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from services.model_adapter.adapter import MockAdapter, validate_request

ROOT = Path(__file__).resolve().parents[2]


def make_server(host="127.0.0.1", port=8000):
    jobs = {}
    adapter = MockAdapter()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, code, message):
            self.respond(status, {"error": {"code": code, "message": message}})

        def do_GET(self):
            if self.path == "/":
                data = (ROOT / "apps" / "frontend" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/health":
                self.respond(200, {"status": "ok", "engine": "mock-v1"})
            elif self.path.startswith("/api/v1/jobs/"):
                job = jobs.get(self.path.removeprefix("/api/v1/jobs/"))
                if job is None:
                    self.error(404, "not_found", "Unknown job")
                else:
                    self.respond(200, job)
            else:
                self.error(404, "not_found", "Unknown route")

        def do_POST(self):
            if self.path != "/api/v1/jobs":
                self.error(404, "not_found", "Unknown route")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Request must be between 1 and 65536 bytes")
                request = json.loads(self.rfile.read(size))
                validate_request(request)
            except (ValueError, UnicodeDecodeError) as error:
                self.error(400, "invalid_request", str(error))
                return
            try:
                result = adapter.predict(request)
            except Exception:
                self.error(500, "prediction_failed", "Model adapter failed")
                return
            job_id = uuid4().hex
            job = {"job_id": job_id, "status": "succeeded", "result": result}
            jobs[job_id] = job
            self.respond(201, job)

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    print("Mock development server: http://127.0.0.1:8000", flush=True)
    with make_server() as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
