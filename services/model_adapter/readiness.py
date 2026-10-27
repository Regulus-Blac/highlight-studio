"""Offline asset admission reports; pointers and unknown provenance are not ready."""

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re

from services.media.probe import digest_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = Path(__file__).with_name("manifests") / "f1-expected-assets.json"


def audit_assets(asset_root, inventory_path=DEFAULT_INVENTORY):
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    root = Path(asset_root).resolve()
    results = []
    names = set()
    for item in inventory["artifacts"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in item["path"] or item["path"] in names:
            raise ValueError("Unsafe or duplicate expected asset path")
        names.add(item["path"])
        expected = item.get("sha256")
        if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid expected SHA256")
        path = (root / item["path"]).resolve()
        status, actual = "missing", None
        if not path.is_relative_to(root):
            status = "unsafe_path"
        elif path.is_file():
            with path.open("rb") as source:
                pointer = source.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
            if pointer:
                status = "lfs_pointer"
            elif expected is None:
                status = "expected_hash_unknown"
            else:
                actual = digest_file(path)
                status = "verified" if actual == expected and (item.get("bytes") is None or item["bytes"] == path.stat().st_size) else "mismatch"
        results.append({"path": item["path"], "kind": item["kind"], "status": status, "expected_sha256": expected, "actual_sha256": actual})
    lock = json.loads((ROOT / "model.lock.json").read_text())
    blockers = list(inventory["unresolved_requirements"])
    if inventory["competition_commit"] != lock["commit"]:
        blockers.append("competition_commit_mismatch")
    if not inventory["complete"]:
        blockers.append("expected_inventory_incomplete")
    if not all(item["status"] == "verified" for item in results):
        blockers.append("assets_not_verified")
    return {"schema_version": 1, "release_id": inventory["release_id"], "competition_commit": inventory["competition_commit"],
            "declared_parameters": inventory["declared_parameters"], "ready": not blockers,
            "counts": dict(Counter(item["status"] for item in results)), "assets": results, "blockers": blockers,
            "models_loaded": False, "gpu_verified": False}
