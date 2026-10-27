"""Verify already downloaded model files; never downloads or executes models."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.model_adapter.assets import read_manifest, verify_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--require-inference", action="store_true")
    args = parser.parse_args()
    commit = json.loads((ROOT / "model.lock.json").read_text(encoding="utf-8"))["commit"]
    try:
        manifest = read_manifest(args.manifest)
        checked = verify_assets(manifest, args.root, competition_commit=commit, require_complete=args.require_inference)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"release_id": manifest["release_id"], "integrity": "verified", "inference_release_declared_complete": manifest["complete"], "competition_commit_matches": manifest.get("competition_commit") == commit, "artifacts": checked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
