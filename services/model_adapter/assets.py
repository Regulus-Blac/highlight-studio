"""Local asset verification only; this module never downloads weights."""

import hashlib
import json
from pathlib import Path, PureWindowsPath
import re


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("prediction_api_version") != "1":
        raise ValueError("Unsupported model manifest version")
    if not isinstance(manifest.get("release_id"), str) or not manifest["release_id"].strip():
        raise ValueError("Model release_id is required")
    if type(manifest.get("complete")) is not bool:
        raise ValueError("Manifest must explicitly declare complete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Manifest must list model artifacts")
    names, paths = set(), set()
    for entry in artifacts:
        name, relative, digest = entry["name"], entry["path"], entry["sha256"]
        p = Path(relative)
        if not name or name in names or relative in paths:
            raise ValueError("Artifact names and paths must be unique")
        if p.is_absolute() or PureWindowsPath(relative).drive or ".." in p.parts or "\\" in relative:
            raise ValueError("Artifact paths must be relative POSIX paths without traversal")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Artifact SHA256 must contain 64 lowercase hexadecimal characters")
        names.add(name)
        paths.add(relative)
    return manifest


def verify_assets(manifest, root, competition_commit=None, require_complete=False):
    root = Path(root).resolve()
    if require_complete and not manifest["complete"]:
        raise ValueError("This manifest is a historical/incomplete sample, not an inference release")
    if require_complete and manifest.get("competition_commit") != competition_commit:
        raise ValueError("Model manifest does not match model.lock.json competition commit")
    checked = []
    for entry in manifest["artifacts"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact escapes the model directory")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            first = source.read(1024 * 1024)
            if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"{entry['path']} is an LFS pointer, not a downloaded artifact")
            digest.update(first)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch: {entry['path']}")
        checked.append({"name": entry["name"], "path": entry["path"], "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    return checked
