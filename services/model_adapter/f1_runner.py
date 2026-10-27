"""F1 preparation entry point: offline admission and explicit stage orchestration."""

import argparse
import json
from pathlib import Path
import subprocess

from services.model_adapter.adapter import validate_request, validate_result
from services.model_adapter.assets import read_manifest, verify_assets
from services.model_adapter.readiness import audit_assets, DEFAULT_INVENTORY, ROOT

STAGES = ("extract_features", "select_frames", "predict_crops", "assemble_result")


def run_recipe(request, result_path, recipe_path, asset_root, manifest_path):
    """Execute a supplied, reviewed recipe; this does not invent the missing F1 recipe."""
    validate_request(request)
    manifest = read_manifest(manifest_path)
    verify_assets(manifest, asset_root, competition_commit=json.loads((ROOT / "model.lock.json").read_text())["commit"], require_complete=True)
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe.get("schema_version") != 1 or recipe.get("release_id") != manifest["release_id"] or [s.get("name") for s in recipe.get("stages", [])] != list(STAGES):
        raise ValueError("Recipe must bind the complete release and all four ordered inference stages")
    output = Path(result_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request_path = output.parent / "stage-request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output.unlink(missing_ok=True)
    values = {"request": str(request_path), "result": str(output), "work": str(output.parent), "asset_root": str(Path(asset_root).resolve()), "manifest": str(Path(manifest_path).resolve())}
    for stage in recipe["stages"]:
        command, timeout = stage.get("command"), stage.get("timeout_seconds", 900)
        if not isinstance(command, list) or not command or any(not isinstance(a, str) or not a for a in command) or type(timeout) not in (int, float) or not 0 < timeout <= 86400:
            raise ValueError("Stage requires a command argument array and a bounded timeout")
        with (output.parent / (stage["name"] + ".log")).open("wb") as log:
            subprocess.run([arg.format_map(values) for arg in command], cwd=output.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
    result = json.loads(output.read_text())
    validate_result(result, request)
    if result["mock"] or result["engine"] != manifest["release_id"]:
        output.unlink(missing_ok=True)
        raise ValueError("Recipe must return the pinned real engine identity")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.check:
        report = {**audit_assets(args.asset_root, args.inventory), "stage_order": list(STAGES)}
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report["ready"] else 2)
    if not all((args.recipe, args.manifest, args.request, args.result)):
        parser.error("--run requires --recipe, --manifest, --request and --result; no default F1 recipe exists")
    run_recipe(json.loads(args.request.read_text()), args.result, args.recipe, args.asset_root, args.manifest)


if __name__ == "__main__":
    main()
