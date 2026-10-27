"""Public API and runner-boundary checks; fixtures never execute a real model."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from apps.backend.server import make_server
from services.model_adapter.adapter import AdapterError, MockAdapter
from services.model_adapter.assets import read_manifest, verify_assets
from services.model_adapter.command import CommandAdapter, UnavailableAdapter
from services.task_service.store import TaskStore

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def serve(root, **options):
    server = make_server(port=0, runtime=root, **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None, method=None, headers=None):
        data = payload if isinstance(payload, bytes) else None if payload is None else json.dumps(payload).encode()
        req = Request(base + path, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(req, timeout=5) as response:
                body = response.read()
                return response.status, json.loads(body) if body else None, response.headers
        except HTTPError as error:
            return error.code, json.loads(error.read()), error.headers

    try:
        yield request, server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def upload(request):
    status, result, _ = request("/api/v1/uploads?filename=demo.mp4", b"mock video fixture; not a playable video")
    assert status == 201, result
    return result


def submit(request, item, **options):
    return request("/api/v1/tasks", {"api_version": "1", "upload_id": item["upload_id"], "target_ratio": [9, 16]}, **options)


def wait_task(request, task_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status, task, _ = request("/api/v1/tasks/" + task_id)
        assert status == 200
        if task["status"] in {"succeeded", "failed", "cancelled"}:
            return task
        time.sleep(0.02)
    raise AssertionError("Task did not reach a terminal state")


class TaskAPITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_upload_queue_predict_and_persistent_result(self):
        with serve(self.root) as (request, _):
            item = upload(request)
            self.assertNotIn("path", item)
            self.assertEqual(item["sha256"], hashlib.sha256(b"mock video fixture; not a playable video").hexdigest())
            status, task, _ = submit(request, item)
            self.assertEqual(status, 202)
            self.assertEqual(task["status"], "queued")
            task = wait_task(request, task["task_id"])
            self.assertEqual(task["status"], "succeeded")
            self.assertTrue(task["mock"])
            self.assertEqual(task["progress"], {"stage": "completed", "percent": 100})
            result_status, result, _ = request(f"/api/v1/tasks/{task['task_id']}/result")
            self.assertEqual(result_status, 200)
            self.assertEqual(result, task["result"])
            self.assertNotIn("video_path", json.dumps(task))
        with serve(self.root) as (request, _):
            self.assertEqual(request("/api/v1/tasks/" + task["task_id"])[1], task)

    def test_queued_task_survives_restart_and_can_be_cancelled(self):
        with serve(self.root, start_worker=False) as (request, _):
            item = upload(request)
            _, task, _ = submit(request, item)
            _, cancelled, _ = submit(request, item)
            cancel_path = f"/api/v1/tasks/{cancelled['task_id']}/cancel"
            self.assertEqual(request(cancel_path, method="POST")[1]["status"], "cancelled")
            self.assertEqual(request(cancel_path, method="POST")[0], 200)
            self.assertEqual(request(f"/api/v1/tasks/{task['task_id']}/result")[0], 409)
        with serve(self.root) as (request, _):
            self.assertEqual(wait_task(request, task["task_id"])["status"], "succeeded")
            self.assertEqual(request("/api/v1/tasks/" + cancelled["task_id"])[1]["status"], "cancelled")
            self.assertEqual(request(f"/api/v1/tasks/{task['task_id']}/cancel", method="POST")[0], 409)

    def test_concurrent_idempotent_submission_runs_once(self):
        class CountingAdapter(MockAdapter):
            calls = 0

            def predict(self, request):
                self.calls += 1
                return super().predict(request)

        adapter = CountingAdapter()
        with serve(self.root, adapter=adapter) as (request, _):
            item = upload(request)
            with ThreadPoolExecutor(max_workers=5) as pool:
                responses = list(pool.map(lambda _: submit(request, item, headers={"Idempotency-Key": "same-upload"}), range(5)))
            self.assertEqual(sorted(r[0] for r in responses), [200, 200, 200, 200, 202])
            task_ids = {r[1]["task_id"] for r in responses}
            self.assertEqual(len(task_ids), 1)
            self.assertEqual(wait_task(request, task_ids.pop())["status"], "succeeded")
            self.assertEqual(adapter.calls, 1)
            bad = {"api_version": "1", "upload_id": item["upload_id"], "target_ratio": [16, 9]}
            self.assertEqual(request("/api/v1/tasks", bad, headers={"Idempotency-Key": "same-upload"})[0], 409)

    def test_missing_model_is_failed_not_mock_success(self):
        with serve(self.root, adapter=UnavailableAdapter()) as (request, _):
            self.assertEqual(request("/ready")[0], 503)
            model = request("/api/v1/model")[1]
            self.assertFalse(model["mock"])
            self.assertFalse(model["available"])
            _, task, _ = submit(request, upload(request))
            task = wait_task(request, task["task_id"])
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["error"]["code"], "model_unavailable")
            self.assertIsNone(task["result"])
            self.assertEqual(request("/api/v1/jobs", {})[0], 410)

    def test_interrupted_task_and_pinned_model_are_explicit_failures(self):
        with serve(self.root, start_worker=False) as (request, server):
            item = upload(request)
            _, interrupted, _ = submit(request, item)
            self.assertIsNotNone(server.store.claim())
        with serve(self.root) as (request, _):
            task = request("/api/v1/tasks/" + interrupted["task_id"])[1]
            self.assertEqual(task["error"]["code"], "worker_interrupted")
            _, queued, _ = submit(request, item)
            wait_task(request, queued["task_id"])
        with serve(self.root, start_worker=False) as (request, _):
            _, old, _ = submit(request, item)

        class NewAdapter(MockAdapter):
            engine = "mock-v2"
            identity = "mock-v2"

        with serve(self.root, adapter=NewAdapter()) as (request, _):
            old = wait_task(request, old["task_id"])
            self.assertEqual(old["error"]["code"], "model_version_unavailable")

    def test_invalid_result_is_not_published(self):
        class BrokenAdapter(MockAdapter):
            def predict(self, request):
                result = super().predict(request)
                result["predictions"] = [{"frame": 0, "bboxes": [0, 0, float("nan")]}]
                return result

        with serve(self.root, adapter=BrokenAdapter()) as (request, _):
            _, task, _ = submit(request, upload(request))
            task = wait_task(request, task["task_id"])
            self.assertEqual(task["error"]["code"], "invalid_prediction")
            self.assertIsNone(task["result"])

    def test_bad_requests_upload_limit_and_cors(self):
        with serve(self.root, max_upload_bytes=64, cors_origin="http://localhost:5173") as (request, _):
            self.assertEqual(request("/api/v1/uploads?filename=x.mp4", b"x" * 65)[0], 413)
            self.assertEqual(request("/api/v1/uploads?filename=x.txt", b"x")[0], 415)
            self.assertEqual(request("/api/v1/uploads", b"x")[0], 400)
            self.assertEqual(request("/api/v1/tasks", b"not JSON")[0], 400)
            self.assertEqual(request("/api/v1/tasks", {"api_version": "1", "video_path": "/etc/passwd", "target_ratio": [9, 16]})[0], 400)
            item = upload(request)
            self.assertEqual(request("/api/v1/tasks", {"api_version": "1", "upload_id": item["upload_id"], "target_ratio": [True, 16]})[0], 400)
            self.assertEqual(request("/api/v1/tasks", {"api_version": "1", "upload_id": "0" * 32, "target_ratio": [9, 16]})[0], 404)
            self.assertEqual(request("/api/v1/tasks/unknown")[0], 404)
            status, _, headers = request("/api/v1/tasks", method="OPTIONS", headers={"Origin": "http://localhost:5173"})
            self.assertEqual(status, 204)
            self.assertEqual(headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertEqual(request("/api/v1/tasks", method="OPTIONS", headers={"Origin": "https://unconfigured.invalid"})[0], 403)

    def test_worker_lock_prevents_two_consumers_and_false_recovery(self):
        with serve(self.root) as (request, _):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                make_server(port=0, runtime=self.root)
            self.assertEqual(request("/ready")[0], 200)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="runner with spaces ")
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.weight = self.root / "fixture.bin"
        self.weight.write_bytes(b"not a model: runner transport test only")
        self.manifest = {
            "schema_version": 1, "release_id": "test-runner-v1", "prediction_api_version": "1", "complete": True,
            "competition_commit": json.loads((ROOT / "model.lock.json").read_text())["commit"],
            "artifacts": [{"name": "fixture", "path": "fixture.bin", "sha256": hashlib.sha256(self.weight.read_bytes()).hexdigest()}],
        }
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.runner = self.root / "runner with spaces.py"
        self.runner.write_text("""import json, sys, time
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
mode = sys.argv[3]
if mode == 'sleep': time.sleep(10)
if mode == 'fail': raise SystemExit(7)
result = {'api_version':'1', 'video_id':request['video_id'], 'target_ratio':request['target_ratio'], 'engine':'test-runner-v1', 'mock': mode == 'invalid', 'predictions':[{'frame':0,'bboxes':[0,0,1]}]}
Path(sys.argv[2]).write_text(json.dumps(result))
""")
        self.request = {"api_version": "1", "video_id": "fixture", "video_path": str(self.weight), "target_ratio": [9, 16]}

    def adapter(self, mode="ok", timeout=3):
        config = {"manifest": "manifest.json", "asset_root": ".", "command": [sys.executable, str(self.runner), "{request}", "{result}", mode], "timeout_seconds": timeout}
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return CommandAdapter(path, self.root / "runs")

    def test_runner_request_result_transport_with_spaces_in_paths(self):
        result = self.adapter().predict(self.request)
        self.assertFalse(result["mock"])
        self.assertEqual(result["engine"], "test-runner-v1")
        self.assertEqual(result["video_id"], "fixture")

    def test_runner_failure_timeout_and_invalid_identity(self):
        for mode, timeout, code in [("fail", 3, "inference_failed"), ("sleep", 0.1, "inference_timeout"), ("invalid", 3, "invalid_prediction")]:
            with self.subTest(mode=mode), self.assertRaises(AdapterError) as error:
                self.adapter(mode, timeout).predict(self.request)
            self.assertEqual(error.exception.code, code)

    def test_incomplete_hash_mismatch_and_lfs_pointer_are_rejected(self):
        manifest = read_manifest(self.manifest_path)
        manifest["complete"] = False
        with self.assertRaisesRegex(ValueError, "historical/incomplete"):
            verify_assets(manifest, self.root, require_complete=True)
        self.weight.write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            verify_assets(manifest, self.root)
        self.weight.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 42\n")
        with self.assertRaisesRegex(ValueError, "LFS pointer"):
            verify_assets(manifest, self.root)

    def test_wrong_competition_commit_is_not_deployed(self):
        self.manifest["competition_commit"] = "0" * 40
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "model.lock.json"):
            self.adapter()


if __name__ == "__main__":
    unittest.main()
