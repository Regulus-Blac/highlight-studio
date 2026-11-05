import json
from pathlib import Path
import tempfile
import unittest

from services.model_adapter.command import CommandAdapter
from services.model_adapter.f1_setup import candidate_manifest
from services.model_adapter.f1_stage import one_probability_row, result_for, sample_indices


class VideoCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.meta = {"frame_count": 26, "width": 320, "height": 180, "fps": 10}

    def test_reject_failed_or_misaligned_temporal_output(self):
        path = self.root / "features.jsonl"
        row = {"video_id": "upload", "fps": 10, "width": 320, "height": 180, "n_frames": 26,
               "windows": [[0, 2.6, 3]], "p_sec": [.1, .8, .4]}
        path.write_text(json.dumps(row), encoding="utf-8")
        self.assertEqual(one_probability_row(path, self.meta), row)
        for bad in ({"error": "could not decode"}, {"video_id": "old-contest-video"}, {"n_frames": 25},
                    {"p_sec": [float("nan"), .8, .4]}, {"p_sec": [.1, .8]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                path.write_text(json.dumps({**row, **bad}), encoding="utf-8")
                one_probability_row(path, self.meta)

    def test_fractional_clock_and_original_frame_bounds(self):
        indices = sample_indices(111, 30000 / 1001, 4)
        self.assertEqual(indices[:6], [0, 7, 15, 22, 30, 37])
        self.assertLess(indices[-1], 111)
        request = {"api_version": "1", "video_id": "../literal-product-id", "video_path": "/media/video.mp4", "target_ratio": [9, 16]}
        valid = {"frame": 25, "bboxes": [110, 2, 99]}
        result = result_for(request, self.meta, [valid])
        self.assertEqual(result["video_id"], request["video_id"])
        self.assertEqual(result["predictions"], [valid])
        for bad in ({"frame": 26}, {"bboxes": [222, 2, 99]}, {"bboxes": [110, 5, 99]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                result_for(request, self.meta, [{**valid, **bad}])

    def test_candidate_manifest_is_not_accepted_by_product_worker(self):
        manifest = candidate_manifest()
        self.assertEqual(len(manifest["artifacts"]), 50)
        (self.root / "model-assets.json").write_text(json.dumps(manifest), encoding="utf-8")
        config = self.root / "adapter-config.json"
        config.write_text(json.dumps({"manifest": "model-assets.json", "asset_root": "."}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not an inference release"):
            CommandAdapter(config, self.root / "runs")


if __name__ == "__main__":
    unittest.main()
