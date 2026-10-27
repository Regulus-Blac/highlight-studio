"""Local-only inventory admission and ordered runner protocol checks."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from services.model_adapter.f1_runner import run_recipe, STAGES
from services.model_adapter.readiness import audit_assets, DEFAULT_INVENTORY

ROOT = Path(__file__).resolve().parents[1]


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="asset audit ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.commit = json.loads((ROOT / "model.lock.json").read_text())["commit"]

    def test_complete_status_report_does_not_stop_at_first_missing(self):
        fixtures = {"good.bin": b"verified", "bad.bin": b"corrupt", "pointer.bin": b"version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 100\n", "unknown.bin": b"unknown"}
        for name, data in fixtures.items():
            (self.root / name).write_bytes(data)
        artifacts = [{"path": name, "kind": "fixture", "sha256": None if name == "unknown.bin" else hashlib.sha256(b"verified").hexdigest()} for name in [*fixtures, "missing.bin"]]
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps({"artifacts": artifacts, "unresolved_requirements": [], "competition_commit": self.commit, "release_id": "test", "declared_parameters": 0, "complete": True}))
        report = audit_assets(self.root, inventory)
        self.assertFalse(report["ready"])
        self.assertEqual(report["counts"], {"verified": 1, "mismatch": 1, "lfs_pointer": 1, "expected_hash_unknown": 1, "missing": 1})
        self.assertFalse(report["models_loaded"])

    def test_path_traversal_and_symlink_escape(self):
        inventory = json.loads(DEFAULT_INVENTORY.read_text())
        path = self.root / "inventory.json"
        inventory["artifacts"] = [{"path": "../outside", "kind": "weight", "sha256": None}]
        path.write_text(json.dumps(inventory))
        with self.assertRaises(ValueError):
            audit_assets(self.root, path)
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes(b"no reading outside root")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        try:
            (self.root / "escape").symlink_to(outside)
        except OSError:
            self.skipTest("OS account does not support symlinks")
        inventory["artifacts"][0]["path"] = "escape"
        path.write_text(json.dumps(inventory))
        self.assertEqual(audit_assets(self.root, path)["assets"][0]["status"], "unsafe_path")

    def test_known_f1_inventory_explicitly_blocks_deployment(self):
        report = audit_assets(self.root)
        self.assertFalse(report["ready"])
        self.assertIn("expected_inventory_incomplete", report["blockers"])
        inventory = json.loads(DEFAULT_INVENTORY.read_text())
        self.assertEqual(sum(a["kind"] == "weight" and a["sha256"] is not None for a in inventory["artifacts"]), 16)
        self.assertEqual(report["declared_parameters"], 8926602839)
        result = subprocess.run([sys.executable, "-m", "services.model_adapter.f1_runner", "--check", "--asset-root", str(self.root)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["stage_order"], list(STAGES))

    def test_recipe_executes_ordered_stages_and_validates_result_identity(self):
        asset = self.root / "fixture.bin"
        asset.write_bytes(b"not a weight; protocol fixture")
        manifest = {"schema_version": 1, "release_id": "cpu-protocol-fixture", "prediction_api_version": "1", "competition_commit": self.commit, "complete": True,
                    "artifacts": [{"name": "fixture", "path": asset.name, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}]}
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        script = self.root / "stage.py"
        script.write_text("""import json,sys
from pathlib import Path
work=Path(sys.argv[1]); name=sys.argv[2]
with (work/'order.txt').open('a') as f: f.write(name+'\\n')
if name=='assemble_result':
 r=json.loads(Path(sys.argv[3]).read_text()); r.pop('video_path')
 r.update(engine='cpu-protocol-fixture',mock=False,predictions=[])
 Path(sys.argv[4]).write_text(json.dumps(r))
""")
        recipe = {"schema_version": 1, "release_id": manifest["release_id"], "stages": [{"name": name, "command": [sys.executable, str(script), "{work}", name, "{request}", "{result}"]} for name in STAGES]}
        recipe_path = self.root / "recipe.json"
        recipe_path.write_text(json.dumps(recipe))
        output = self.root / "runs/result.json"
        request = {"api_version": "1", "video_id": "cpu-fixture", "video_path": str(asset), "target_ratio": [16, 9]}
        result = run_recipe(request, output, recipe_path, self.root, manifest_path)
        self.assertFalse(result["mock"])
        self.assertEqual((output.parent / "order.txt").read_text().splitlines(), list(STAGES))
        recipe["stages"].reverse()
        recipe_path.write_text(json.dumps(recipe))
        with self.assertRaises(ValueError):
            run_recipe(request, output, recipe_path, self.root, manifest_path)


if __name__ == "__main__":
    unittest.main()
