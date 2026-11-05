"""F1 preparation entry point: offline admission and explicit stage orchestration."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.model_adapter.adapter import validate_request, validate_result
from services.model_adapter.assets import read_manifest, verify_assets
from services.model_adapter.command import stop_runner
from services.model_adapter.readiness import audit_assets, DEFAULT_INVENTORY, ROOT

STAGES = ("extract_features", "select_frames", "predict_crops", "assemble_result")


def run_recipe(request, result_path, recipe_path, asset_root, manifest_path, *, validate_candidate=False):
    """Execute the recipe; an unapproved F1 candidate runs only via explicit validation."""
    validate_request(request)
    manifest = read_manifest(manifest_path)
    commit = json.loads((ROOT / "model.lock.json").read_text())["commit"]
    if validate_candidate:
        from services.model_adapter.f1_setup import candidate_manifest
        if manifest != candidate_manifest():
            raise ValueError("Validation accepts only the pinned incomplete F1 candidate manifest")
    verify_assets(manifest, asset_root, competition_commit=commit, require_complete=not validate_candidate)
    recipe = json.loads(Path(recipe_path).read_text())
    if recipe.get("schema_version") != 1 or recipe.get("release_id") != manifest["release_id"] or [s.get("name") for s in recipe.get("stages", [])] != list(STAGES):
        raise ValueError("Recipe must bind the complete release and all four ordered inference stages")
    output = Path(result_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    request_path = output.parent / "stage-request.json"
    if output.exists() or output == request_path:
        raise ValueError("Inference requires a fresh result path")
    with request_path.open("x", encoding="utf-8") as stream:
        json.dump(request, stream)
    values = {"request": str(request_path), "result": str(output), "work": str(output.parent), "asset_root": str(Path(asset_root).resolve()), "manifest": str(Path(manifest_path).resolve())}
    for stage in recipe["stages"]:
        command, timeout = stage.get("command"), stage.get("timeout_seconds", 900)
        if not isinstance(command, list) or not command or any(not isinstance(a, str) or not a for a in command) or type(timeout) not in (int, float) or not 0 < timeout <= 86400:
            raise ValueError("Stage requires a command argument array and a bounded timeout")
        with (output.parent / (stage["name"] + ".log")).open("wb") as log:
            argv = [arg.format_map(values) for arg in command]
            kwargs = dict(cwd=output.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            if not validate_candidate:
                subprocess.run(argv, **kwargs, timeout=timeout, check=True)
                continue
            # Standalone validation has no CommandAdapter process group to clean up.
            process = subprocess.Popen(argv, **kwargs, start_new_session=os.name == "posix")
            try:
                if process.wait(timeout=timeout):
                    raise subprocess.CalledProcessError(process.returncode, argv)
            finally:
                if process.poll() is None or process.returncode:
                    stop_runner(process)
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
    parser.add_argument("--validate-candidate", action="store_true", help="Run the pinned candidate explicitly before promotion")
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
    run_recipe(json.loads(args.request.read_text(encoding="utf-8")), args.result, args.recipe, args.asset_root, args.manifest,
               validate_candidate=args.validate_candidate)


if __name__ == "__main__":
    main()
