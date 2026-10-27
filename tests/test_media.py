"""Synthetic, explicitly labelled CPU fixtures; never asserts F1 model quality."""

import array
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from services.media.plan import aligned_crop, interpolate_box, make_plan, seconds_to_frame
from services.media.probe import MediaError, binary, digest_file, probe
from services.media.render import render_jsonl, render_mp4
from services.media.service import MediaService
from services.model_adapter.adapter import MockAdapter
from services.task_service.store import TaskError, TaskStore
from tests.test_integration import serve, wait_task

ROOT = Path(__file__).resolve().parents[1]
META = {"width": 192, "height": 108, "fps_num": 10, "fps_den": 1, "frame_count": 30, "duration_seconds": 3.0,
        "has_audio": True, "rotation": 0, "validation_profile": "cpu-cfr-v1"}


def prediction(video_id="synthetic"):
    return {"api_version": "1", "video_id": video_id, "target_ratio": [16, 9], "engine": "synthetic-cpu-v1", "mock": True,
            "predictions": [{"frame": frame, "bboxes": [0, 0, 96] if frame < 20 else [96, 0, 64]} for frame in [*range(2, 7), *range(20, 25)]]}


class FixtureAdapter(MockAdapter):
    engine = identity = "synthetic-cpu-v1"

    def predict(self, request):
        return prediction(request["video_id"])


def validate_contract(name, instance):
    schemas = {p.name: json.loads(p.read_text()) for p in (ROOT / "contracts").glob("*.schema.json")}
    registry = Registry().with_resources((name, Resource.from_contents(schema)) for name, schema in schemas.items())
    Draft202012Validator(schemas[name + ".schema.json"], registry=registry).validate(instance)


class PlanTests(unittest.TestCase):
    def test_frozen_schema_examples(self):
        for schema in (ROOT / "contracts").glob("*.schema.json"):
            Draft202012Validator.check_schema(json.loads(schema.read_text()))
        for example in (ROOT / "contracts/examples").glob("*.json"):
            with self.subTest(example=example.name):
                validate_contract(example.stem, json.loads(example.read_text()))

    def test_contiguous_runs_single_frames_no_gap_fill_and_empty(self):
        item = prediction()
        item["predictions"] = [{"frame": i, "bboxes": [0, 0, 192]} for i in (0, 1, 5, 29)]
        plan = make_plan(item, META)
        self.assertEqual([(s["start_frame"], s["end_frame_exclusive"]) for s in plan["segments"]], [(0, 2), (5, 6), (29, 30)])
        self.assertEqual([f["frame"] for f in plan["frame_crops"]], [0, 1, 5, 29])
        self.assertAlmostEqual(plan["duration_seconds"], 0.4)
        item["predictions"] = []
        self.assertIsNone(make_plan(item, META)["output"])

    def test_bounds_nonfinite_duplicates_and_bad_ratio_are_rejected(self):
        for frame, box in [(30, [0, 0, 192]), (-1, [0, 0, 192]), (2, [1, 0, 192]), (2, [0, 55, 96]), (2, [0, 0, float("nan")])]:
            with self.subTest(frame=frame, box=box), self.assertRaises(MediaError):
                item = prediction()
                item["predictions"] = [{"frame": frame, "bboxes": box}]
                make_plan(item, META)
        item = prediction()
        item["predictions"] *= 2
        with self.assertRaises(MediaError):
            make_plan(item, META)

    def test_edit_extension_uses_own_suggestion_edge_and_snapshot_identity(self):
        original = prediction()
        before = deepcopy(original)
        segments = [{"suggestion_id": "0", "start_frame": 0, "end_frame_exclusive": 9, "enabled": True},
                    {"suggestion_id": "1", "start_frame": 25, "end_frame_exclusive": 30, "enabled": True}]
        plan = make_plan(original, META, segments)
        self.assertEqual(plan["frame_crops"][0]["crop"], [0, 0, 96, 54])
        self.assertEqual(plan["frame_crops"][-1]["crop"], [96, 0, 64, 36])
        self.assertEqual(original, before)
        self.assertNotEqual(plan["plan_id"], make_plan(original, META)["plan_id"])
        self.assertEqual(plan["plan_id"], make_plan(original, META, segments)["plan_id"])

    def test_edits_reject_overlap_unknown_duplicate_and_out_of_bounds(self):
        valid = make_plan(prediction(), META)["segments"]
        mutations = [lambda s: s[1].update(start_frame=5), lambda s: s[0].update(suggestion_id="unknown"),
                     lambda s: s[1].update(suggestion_id="0"), lambda s: s[0].update(start_frame=-1),
                     lambda s: s[0].update(end_frame_exclusive=2), lambda s: s.reverse()]
        for mutate in mutations:
            edited = deepcopy(valid)
            mutate(edited)
            with self.subTest(edited=edited), self.assertRaises(MediaError):
                make_plan(prediction(), META, edited)

    def test_inward_alignment_ratio_common_size_and_rounding(self):
        crop = aligned_crop([0.25, 1.25, 190.5], [16, 9])
        self.assertEqual(crop, [1, 2, 160, 90])
        self.assertEqual(make_plan(prediction(), META)["output"], {"width": 64, "height": 36})
        self.assertEqual(seconds_to_frame("0.05", META), 1)
        self.assertEqual(seconds_to_frame("0.049", META), 0)
        self.assertEqual(interpolate_box({0: [0, 0, 100], 10: [10, 20, 80]}, 5), [5, 10, 90])
        with self.assertRaises(MediaError):
            aligned_crop([0, 0, 1], [9, 16])


class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = tempfile.TemporaryDirectory(prefix="cpu media fixtures ")
        cls.addClassCleanup(cls.fixtures.cleanup)
        cls.source = Path(cls.fixtures.name) / "audio source.mp4"
        cls.silent = Path(cls.fixtures.name) / "silent.mp4"
        subprocess.run([binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=red:size=192x108:rate=10:duration=3,drawbox=x=96:y=0:w=96:h=108:color=blue:t=fill",
                        "-f", "lavfi", "-i", "aevalsrc=sin(2*PI*if(lt(t\\,1)\\,440\\,880)*t):s=48000:d=3",
                        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", "-y", str(cls.source)], check=True)
        subprocess.run([binary("ffmpeg"), "-v", "error", "-i", str(cls.source), "-c:v", "copy", "-an", "-y", str(cls.silent)], check=True)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="media test ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def seed(self, *, allow_mock=True):
        store = TaskStore(self.root)
        upload_id = uuid4().hex
        source = store.uploads / (upload_id + ".mp4")
        shutil.copyfile(self.source, source)
        store.add_upload(upload_id, "fixture.mp4", source, source.stat().st_size, digest_file(source))
        task, _ = store.create_task({"api_version": "1", "upload_id": upload_id, "target_ratio": [16, 9]}, FixtureAdapter())
        store.claim()
        store.finish(task["task_id"], prediction(upload_id))
        return store, MediaService(store, allow_mock=allow_mock), task["task_id"], source

    def test_probe_and_jsonl_schema(self):
        meta = probe(self.source)
        self.assertEqual(meta, META)
        result = prediction()
        plan = make_plan(result, meta)
        path = self.root / "out.jsonl"
        render_jsonl(result, plan, path)
        rows = path.read_text().splitlines()
        self.assertEqual(len(rows), 1)
        validate_contract("prediction-result", json.loads(rows[0]))

    def test_explicit_development_adapter_and_portrait_mp4(self):
        from services.model_adapter.command import create_adapter
        adapter = create_adapter("mock-media")
        request = {"api_version": "1", "video_id": "synthetic", "video_path": str(self.source), "target_ratio": [9, 16]}
        result = adapter.predict(request)
        self.assertTrue(result["mock"])
        self.assertEqual(result["engine"], "synthetic-media-v1")
        plan = make_plan(result, probe(self.source))
        actual = render_mp4(self.source, plan, self.root / "portrait.mp4")
        self.assertEqual((actual["width"], actual["height"]), (54, 96))
        self.assertEqual(actual["frame_count"], 14)
        self.assertTrue(actual["has_audio"])

    def test_mp4_exact_frames_dynamic_crop_and_audio_intervals(self):
        output = self.root / "out.mp4"
        plan = make_plan(prediction(), probe(self.source))
        actual = render_mp4(self.source, plan, output)
        self.assertEqual(actual["frame_count"], 10)
        self.assertEqual((actual["width"], actual["height"]), (64, 36))
        self.assertTrue(actual["has_audio"])
        pixels = subprocess.check_output([binary("ffmpeg"), "-v", "error", "-i", str(output), "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        stride = 64 * 36 * 3
        for i in range(10):
            frame = pixels[i * stride:(i + 1) * stride]
            red, blue = sum(frame[0::3]) / (64 * 36), sum(frame[2::3]) / (64 * 36)
            self.assertGreater(red - blue if i < 5 else blue - red, 180)
        samples = array.array("f", subprocess.check_output([binary("ffmpeg"), "-v", "error", "-i", str(output), "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"]))
        for start, end, frequency in [(0.1, 0.4, 440), (0.6, 0.9, 880)]:
            chunk = samples[int(start * 48000):int(end * 48000)]
            observed = sum(a < 0 <= b for a, b in zip(chunk, chunk[1:])) / (end - start)
            self.assertAlmostEqual(observed, frequency, delta=15)
        audio = subprocess.check_output([binary("ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "json", str(output)])
        self.assertAlmostEqual(float(json.loads(audio)["streams"][0]["duration"]), 1, delta=0.05)

    def test_silent_single_frame_and_empty_mp4(self):
        item = prediction()
        item["predictions"] = item["predictions"][:1]
        plan = make_plan(item, probe(self.silent))
        actual = render_mp4(self.silent, plan, self.root / "one.mp4")
        self.assertEqual(actual["frame_count"], 1)
        self.assertFalse(actual["has_audio"])
        item["predictions"] = []
        with self.assertRaisesRegex(MediaError, "at least one"):
            render_mp4(self.silent, make_plan(item, probe(self.silent)), self.root / "empty.mp4")

    @unittest.skipIf(sys.platform == "win32", "Unix executable fixture; real media tests run on both platforms")
    def test_renderer_timeout_and_shutdown_kill_blocked_subprocesses(self):
        script = self.root / "blocked-ffmpeg"
        script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
        script.chmod(0o755)
        plan = make_plan(prediction(), META)
        cancelled = threading.Event()
        cancelled.set()
        for settings, expected in [({"timeout": 0.1}, "render_timeout"), ({"cancel_event": cancelled}, "render_interrupted")]:
            with self.subTest(code=expected), patch("services.media.render.binary", return_value=str(script)), self.assertRaises(MediaError) as error:
                render_mp4(self.source, plan, self.root / "blocked.mp4", **settings)
            self.assertEqual(error.exception.code, expected)

    def test_invalid_video_rotation_and_vfr_are_explicit(self):
        broken = self.root / "bad.mp4"
        broken.write_bytes(b"not playable")
        with self.assertRaises(MediaError):
            probe(broken)
        rotated = self.root / "rotated.mp4"
        subprocess.run([binary("ffmpeg"), "-v", "error", "-display_rotation", "90", "-i", str(self.source), "-c", "copy", str(rotated)], check=True)
        with self.assertRaisesRegex(MediaError, "rotation"):
            probe(rotated)
        vfr = self.root / "vfr.mp4"
        subprocess.run([binary("ffmpeg"), "-v", "error", "-i", str(self.silent), "-vf", "setpts=if(lt(N\\,10)\\,N\\,(N+2))/(10*TB)", "-fps_mode", "vfr", "-c:v", "libx264", "-threads", "1", str(vfr)], check=True)
        with self.assertRaisesRegex(MediaError, "Variable"):
            probe(vfr)

    def test_export_snapshot_idempotency_persistence_and_source_tamper(self):
        store, media, task_id, source = self.seed()
        payload = {"api_version": "1", "task_id": task_id, "format": "jsonl"}
        item, created = media.create(payload, "same-export")
        self.assertTrue(created)
        self.assertFalse(media.create(payload, "same-export")[1])
        with self.assertRaises(TaskError):
            media.create({**payload, "format": "mp4"}, "same-export")
        with self.assertRaises(TaskError):
            media.create(payload)
        with store.connect() as db:
            frozen = json.loads(db.execute("SELECT snapshot FROM exports").fetchone()[0])
        self.assertEqual(frozen["plan"]["plan_id"], item["plan_id"])
        media.run_once()
        complete = MediaService(TaskStore(self.root), allow_mock=True).get(item["export_id"])
        self.assertEqual(complete["status"], "succeeded")
        validate_contract("export", complete)
        self.assertEqual(media.get(item["export_id"])["artifact"]["sha256"], digest_file(media.artifact_path(item["export_id"])[0]))
        second, _ = media.create(payload)
        source.write_bytes(b"changed source")
        media.run_once()
        self.assertEqual(media.get(second["export_id"])["error"]["code"], "source_changed")
        self.assertEqual(media.create(payload, "same-export")[0], complete)

    def test_mock_export_gate_and_recovery(self):
        store, media, task_id, _ = self.seed(allow_mock=False)
        payload = {"api_version": "1", "task_id": task_id, "format": "mp4"}
        with self.assertRaisesRegex(TaskError, "development-only"):
            media.create(payload)
        media.allow_mock = True
        item, _ = media.create(payload)
        with store.connect() as db:
            db.execute("UPDATE exports SET status='running'")
        media.recover()
        failed = media.get(item["export_id"])
        self.assertEqual(failed["error"]["code"], "worker_interrupted")
        validate_contract("export", failed)

    def test_http_preview_export_range_and_existing_v1(self):
        with serve(self.root, adapter=FixtureAdapter(), allow_mock_media=True) as (request, server):
            status, item, _ = request("/api/v1/uploads?filename=fixture.mp4", self.source.read_bytes())
            self.assertEqual(status, 201)
            status, metadata, _ = request(f"/api/v1/uploads/{item['upload_id']}/metadata")
            self.assertEqual(status, 200)
            validate_contract("media-metadata", metadata)
            status, task, _ = request("/api/v1/tasks", {"api_version": "1", "upload_id": item["upload_id"], "target_ratio": [16, 9]})
            self.assertEqual(status, 202)
            self.assertEqual(wait_task(request, task["task_id"])["status"], "succeeded")
            status, preview, _ = request(f"/api/v1/tasks/{task['task_id']}/preview")
            self.assertEqual(status, 200)
            validate_contract("preview", preview)
            self.assertNotIn(str(self.root), json.dumps(preview))
            edited = {"api_version": "1", "segments": [preview["segments"][0]]}
            validate_contract("preview-request", edited)
            status, edited_preview, _ = request(f"/api/v1/tasks/{task['task_id']}/preview", edited)
            self.assertEqual(status, 200)
            self.assertNotEqual(edited_preview["plan_id"], preview["plan_id"])
            payload = {"api_version": "1", "task_id": task["task_id"], "format": "mp4"}
            validate_contract("export-request", payload)
            status, export, _ = request("/api/v1/exports", payload, headers={"Idempotency-Key": "http-export"})
            self.assertEqual(status, 202)
            validate_contract("export", export)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                export = request("/api/v1/exports/" + export["export_id"])[1]
                if export["status"] in {"succeeded", "failed"}:
                    break
                time.sleep(0.05)
            self.assertEqual(export["status"], "succeeded", export)
            base = f"http://127.0.0.1:{server.server_port}"
            for route in (preview["source_url"], export["artifact"]["url"]):
                with urlopen(base + route) as response:
                    data = response.read()
                for range_header, expected in [("bytes=0-15", data[:16]), ("bytes=-8", data[-8:]), ("bytes=10-", data[10:])]:
                    with urlopen(Request(base + route, headers={"Range": range_header})) as response:
                        self.assertEqual(response.status, 206)
                        self.assertEqual(response.read(), expected)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + route, headers={"Range": "bytes=999999999-"}))
                self.assertEqual(error.exception.code, 416)
            self.assertEqual(request("/api/v1/exports/not-a-path/file")[0], 404)
            self.assertEqual(request("/api/v1/exports", {**payload, "segments": "bad"})[0], 422)

    def test_real_worker_rejects_bad_media_and_boxes_before_publication(self):
        class RealBoundaryFixture(FixtureAdapter):
            mock = False

            def predict(self, request):
                item = super().predict(request)
                item["mock"] = False
                item["predictions"][0]["bboxes"][2] = 1000
                return item

        with serve(self.root, adapter=RealBoundaryFixture()) as (request, _):
            _, uploaded, _ = request("/api/v1/uploads?filename=fixture.mp4", self.source.read_bytes())
            _, task, _ = request("/api/v1/tasks", {"api_version": "1", "upload_id": uploaded["upload_id"], "target_ratio": [16, 9]})
            failed = wait_task(request, task["task_id"])
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["error"]["code"], "invalid_prediction")
            self.assertIsNone(failed["result"])


if __name__ == "__main__":
    unittest.main()
