"""Bridge the v1 prediction contract to an independently deployed GPU runner."""

import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from uuid import uuid4

from .adapter import AdapterError, MockAdapter, validate_request, validate_result
from .assets import read_manifest, verify_assets

ROOT = Path(__file__).resolve().parents[2]


class UnavailableAdapter:
    engine = "competition-unconfigured"
    mock = False
    identity = "competition-unconfigured"
    available = False

    def predict(self, request):
        raise AdapterError("model_unavailable", "No verified model release and inference runner are configured")


class CommandAdapter:
    mock = False
    available = True

    def __init__(self, config_path, work_root):
        config_path = Path(config_path).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.manifest_path = (config_path.parent / config["manifest"]).resolve()
        self.asset_root = (config_path.parent / config["asset_root"]).resolve()
        manifest = read_manifest(self.manifest_path)
        commit = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))["commit"]
        verify_assets(manifest, self.asset_root, competition_commit=commit, require_complete=True)
        self.engine = manifest["release_id"]
        self.command = config["command"]
        if not isinstance(self.command, list) or not self.command or any(not isinstance(x, str) or not x for x in self.command):
            raise ValueError("Runner command must be a non-empty argument array")
        if not all(any(token in arg for arg in self.command) for token in ("{request}", "{result}")):
            raise ValueError("Runner command requires {request} and {result} placeholders")
        self.timeout = config.get("timeout_seconds", 900)
        if type(self.timeout) not in (int, float) or not 0 < self.timeout <= 86400:
            raise ValueError("timeout_seconds must be between 0 and 86400")
        identity = {"config": config, "manifest": manifest}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.work_root = Path(work_root).resolve()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.closed = threading.Event()

    def close(self):
        self.closed.set()

    def predict(self, request):
        validate_request(request)
        directory = self.work_root / uuid4().hex
        directory.mkdir()
        request_path, result_path = directory / "request.json", directory / "result.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        values = {"{request}": str(request_path), "{result}": str(result_path), "{asset_root}": str(self.asset_root), "{manifest}": str(self.manifest_path)}
        command = list(self.command)
        for token, value in values.items():
            command = [arg.replace(token, value) for arg in command]
        if self.closed.is_set():
            raise AdapterError("inference_interrupted", "Inference worker is stopping", True)
        with (directory / "runner.log").open("wb") as log:
            try:
                process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
            except OSError as error:
                raise AdapterError("runner_unavailable", "Inference runner could not be started") from error
            deadline = time.monotonic() + self.timeout
            try:
                while process.poll() is None:
                    if self.closed.wait(0.05):
                        raise AdapterError("inference_interrupted", "Inference worker is stopping", True)
                    if time.monotonic() >= deadline:
                        raise AdapterError("inference_timeout", "Inference exceeded its configured time limit", True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if process.returncode:
            raise AdapterError("inference_failed", "Inference runner failed; see the worker's local runner.log")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            validate_result(result, request)
            if result["mock"] or result["engine"] != self.engine:
                raise ValueError("Runner must identify the configured real model release")
        except (OSError, ValueError, TypeError) as error:
            raise AdapterError("invalid_prediction", "Inference runner returned an invalid v1 prediction") from error
        return result


def create_adapter(mode="mock", config_path=None, work_root=None):
    if mode == "mock":
        return MockAdapter()
    if mode == "mock-media":
        from services.model_adapter.development import SyntheticMediaAdapter
        return SyntheticMediaAdapter()
    if mode != "command":
        raise ValueError("Adapter must be mock, mock-media or command")
    if not config_path:
        return UnavailableAdapter()
    return CommandAdapter(config_path, work_root or ROOT / "runtime" / "inference")
