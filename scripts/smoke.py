"""Exercise real HTTP requests against the mock product contract on a free port."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apps.backend.server import make_server


def main():
    runtime = tempfile.TemporaryDirectory()
    server = make_server(port=0, runtime=runtime.name)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    try:
        status, body = request("/")
        assert status == 200 and b"Highlight Studio" in body
        status, body = request("/health")
        assert status == 200 and json.loads(body)["engine"] == "mock-v1"
        payload = {"api_version": "1", "video_id": "demo", "video_path": "not-a-real-file.mp4", "target_ratio": [9, 16]}
        status, body = request("/api/v1/jobs", payload)
        job = json.loads(body)
        assert status == 201 and job["status"] == "succeeded"
        result = job["result"]
        assert result == {"api_version": "1", "video_id": "demo", "target_ratio": [9, 16], "engine": "mock-v1", "mock": True, "predictions": []}
        status, body = request("/api/v1/jobs/" + job["job_id"])
        assert status == 200 and json.loads(body) == job
        assert request("/api/v1/jobs/missing")[0] == 404
        assert request("/api/v1/jobs", {**payload, "target_ratio": [0, 16]})[0] == 400
        assert request("/api/v1/jobs", {**payload, "target_ratio": [True, 16]})[0] == 400
        assert request("/api/v1/jobs", [1, 2])[0] == 400
        print("PASS mock HTTP contract: page, health, create/query job, unknown job, invalid requests")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        runtime.cleanup()


if __name__ == "__main__":
    main()
