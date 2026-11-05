"""Prepare a pinned F1-derived video candidate without loading models."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys

from services.media.probe import digest_file
from services.model_adapter.assets import verify_assets
from services.model_adapter.f1_runner import STAGES
from services.model_adapter.readiness import ROOT

RELEASE = "f1-derived-video-v0"
SOURCE_MANIFEST_SHA256 = "b1d3dad539c17f2aece69902a9658e05add9a3d0581e0e06033b0c70b25efc80"
INVENTORY = Path(__file__).parent / "manifests" / "f1-downloaded-assets.json"


def verify_source(source):
    source = Path(source).resolve(strict=True)
    manifest = source / "SOURCE_MANIFEST.json"
    if digest_file(manifest) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Expected the fixed F1 source reference manifest from PR #4")
    for item in json.loads(manifest.read_bytes())["files"]:
        path = (source / item["path"]).resolve(strict=True)
        if not path.is_relative_to(source) or path.stat().st_size != item["bytes"] or digest_file(path) != item["sha256"]:
            raise ValueError(f"F1 source does not match the reference: {item['path']}")
    return source


def candidate_manifest():
    inventory = json.loads(INVENTORY.read_bytes())
    lock = json.loads((ROOT / "model.lock.json").read_bytes())
    if lock["commit"] != inventory["competition_commit"]:
        raise ValueError("F1 asset inventory does not match the product's competition revision")
    return {"schema_version": 1, "prediction_api_version": "1", "release_id": RELEASE,
            "competition_commit": lock["commit"], "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "complete": False,
            "artifacts": [{"name": x["path"], "path": x["path"], "sha256": x["sha256"]}
                          for x in inventory["artifacts"]]}


def materialize(receipt, asset_root, allowed_root):
    """Create a separate hardlink layout on the existing asset filesystem."""
    expected = {a["path"]: a["sha256"] for a in candidate_manifest()["artifacts"]}
    rows = json.loads(Path(receipt).read_bytes())["artifacts"]
    if len(rows) != len(expected) or {r["artifact"]: r["sha256"] for r in rows} != expected:
        raise ValueError("Receipt must bind every artifact in the fixed inventory")
    allowed_root = Path(allowed_root).resolve(strict=True)
    bindings = []
    for row in rows:
        source = Path(row["remote_path"]).resolve(strict=True)
        if row["status"] != "verified" or not source.is_relative_to(allowed_root) or not source.is_file():
            raise ValueError(f"Invalid source binding: {row['artifact']}")
        bindings.append((source, PurePosixPath(row["artifact"])))
    root = Path(asset_root)
    root.mkdir(parents=True, exist_ok=False)
    for source, relative in bindings:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    return root


def setup(source, assets, python, output):
    source = verify_source(source)
    assets = Path(assets).resolve(strict=True)
    python = Path(python).absolute()  # resolving a venv's symlink would select its base interpreter
    if not python.is_file():
        raise ValueError("Inference Python does not exist")
    manifest = candidate_manifest()
    verify_assets(manifest, assets)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner = ROOT / "services" / "model_adapter" / "f1_runner.py"
    stage = runner.with_name("f1_stage.py")
    recipe = {"schema_version": 1, "release_id": RELEASE, "stages": [
        {"name": name, "command": [str(python), str(stage), name, "--source-dir", str(source),
         "--assets", "{asset_root}", "--request", "{request}", "--result", "{result}", "--work", "{work}"],
         "timeout_seconds": limit}
        for name, limit in zip(STAGES, (900, 420, 1200, 60), strict=True)]}
    config = {"manifest": "model-assets.json", "asset_root": str(assets),
              "command": [str(python), str(runner), "--run", "--asset-root", "{asset_root}", "--manifest", "{manifest}",
                          "--recipe", str(output / "recipe.json"), "--request", "{request}", "--result", "{result}"],
              "timeout_seconds": 2700}
    for name, payload in (("model-assets.json", manifest), ("recipe.json", recipe), ("adapter-config.json", config)):
        (output / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, help="Create a new asset root by hardlinking the verified files")
    parser.add_argument("--allowed-root", type=Path, help="Allow hardlink sources only below this filesystem root")
    args = parser.parse_args()
    if args.receipt:
        if not args.allowed_root:
            parser.error("--receipt requires --allowed-root")
        verify_source(args.source_dir)
        materialize(args.receipt, args.asset_root, args.allowed_root)
    output = setup(args.source_dir, args.asset_root, args.python, args.output_dir)
    print(f"Candidate configuration saved to {output}; adapter remains disabled pending GPU validation")


if __name__ == "__main__":
    main()
