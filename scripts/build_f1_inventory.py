"""Build the known F1 asset inventory from locked, small upstream metadata only."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "vendor/competition/resources/team_model_materials_20261023/manifest.json"
    manifest = json.loads(source.read_text())
    artifacts = [{key: item[key] for key in ("path", "kind", "bytes", "sha256")} for item in manifest["files"] if item["kind"] in {"weight", "original_config"}]
    # Keep all original configs even when the upstream kind label changes.
    selected = {a["path"] for a in artifacts}
    for item in manifest["files"]:
        if item["path"].endswith(("config.json", "head.json")) and item["path"] not in selected:
            artifacts.append({key: item[key] for key in ("path", "kind", "bytes", "sha256")})
    unknown = [
        ("timelens/model.safetensors.index.json", "backbone_index"),
        ("timelens/config.json", "backbone_config"),
        ("timelens/preprocessor_config.json", "processor_config"),
        ("timelens/tokenizer_config.json", "tokenizer_config"),
        ("parent/f1-parameter-inventory.json", "complete_parent_inventory"),
        ("parent/temporal-ensemble.json", "frozen_temporal_recipe"),
        ("parent/decoder-trim.json", "frozen_decoder_and_postprocess"),
        ("detectors/yolo11s.pt", "weight"),
        ("environment/lock.json", "inference_environment"),
        ("recipe/f1-new-video.json", "arbitrary_video_runtime_recipe"),
    ]
    artifacts.extend({"path": path, "kind": kind, "sha256": None, "bytes": None} for path, kind in unknown)
    inventory = {"schema_version": 1, "release_id": "f1-43.37-preparation-v1", "complete": False,
        "competition_commit": json.loads((ROOT / "model.lock.json").read_text())["commit"], "declared_parameters": 8926602839,
        "source": "resources/team_model_materials_20261023/manifest.json and docs/F1_FULL2D_REPRO.md at the locked competition commit",
        "path_policy": "Known files use the supplemental package layout. Unknown paths are proposed release slots, not discovered originals.",
        "artifacts": artifacts,
        "unresolved_requirements": [
            "Resolve complete parent inventory (G1/X1 requirements declare 30 weight files; F1 adds one); enumerate all backbone shards and inherited temporal heads with original hashes, configs and parameter counts",
            "Pin TimeLens source revision, all shards, tokenizer and processor; no assumption that old HF feature caches contain the backbone",
            "Recover exact temporal ensemble, decoder shift, selection and trim settings; inherited historical predictions cannot serve new video input",
            "Bind the arbitrary-video feature sampler and 448/672 spatial path to frozen G1/X1/F1 code and configs",
            "Supply locked inference environment and a verified new-video runtime recipe; tools/run_final.py is not F1",
            "GPU end-to-end acceptance and product distribution/source-license review remain pending"
        ]}
    target = ROOT / "services/model_adapter/manifests/f1-expected-assets.json"
    target.write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(artifacts)} asset slots; explicitly incomplete, not a deployable release")


if __name__ == "__main__":
    main()
