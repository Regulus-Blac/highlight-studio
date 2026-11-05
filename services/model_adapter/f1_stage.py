"""Run F1-derived inference on one product video, using no historical predictions."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.media.probe import digest_file, probe
from services.model_adapter.adapter import validate_request, validate_result
from services.model_adapter.f1_runner import STAGES
from services.model_adapter.f1_setup import RELEASE, verify_source

ID = "upload"  # Never use a request ID as an upstream filename.
GROUPS = ("vh_restore_baseline_L1", "vh_restore_baseline_L2", "vh_pl2_L1", "vh_pl2_L2")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def one_probability_row(path, meta):
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len(rows) != 1 or rows[0].get("video_id") != ID or "error" in rows[0]:
        raise ValueError(f"Temporal model failed to return one successful row: {path.name}")
    row = rows[0]
    windows, scores = row.get("windows", []), row.get("p_sec", [])
    if (not windows or not scores or any(type(k) is not int or k <= 0 or not 0 <= a < b for a, b, k in windows)
            or sum(w[2] for w in windows) != len(scores) or any(not 0 <= v <= 1 for v in scores)):
        raise ValueError(f"Invalid temporal windows or probabilities: {path.name}")
    if ((row["n_frames"], row["width"], row["height"]) != (meta["frame_count"], meta["width"], meta["height"])
            or not math.isclose(row["fps"], meta["fps"], abs_tol=0.001, rel_tol=0)):
        raise ValueError("Temporal decoder disagrees with the source video frame clock")
    return row


def sample_indices(n, fps, rate):
    return sorted({round(k * fps / rate) for k in range(math.floor((n - 1) / fps * rate) + 1)})


def result_for(request, meta, predictions):
    result = {"api_version": "1", "video_id": request["video_id"], "target_ratio": request["target_ratio"],
              "engine": RELEASE, "mock": False, "predictions": predictions}
    validate_result(result, request)
    tw, th = request["target_ratio"]
    for item in predictions:
        x, y, width = item["bboxes"]
        if item["frame"] >= meta["frame_count"] or x + width > meta["width"] or y + width * th / tw > meta["height"] + 1e-9:
            raise ValueError("Model output is outside the original video")
    return result


class Run:
    def __init__(self, args):
        self.work, self.result = args.work, args.result
        self.request = read_json(args.request)
        validate_request(self.request)
        if self.request["target_ratio"] not in ([16, 9], [9, 16]):
            raise ValueError("F1 candidate supports 16:9 and 9:16 only")
        self.video = Path(self.request["video_path"])
        if not self.video.is_absolute() or not self.video.is_file():
            raise ValueError("Prediction requires an existing absolute path to the uploaded video")
        source = verify_source(args.source_dir)
        self.sym = source / "historical" / "git_sym8a18"
        self.cot = source / "historical" / "git_cot_a480"
        self.src = source / "src"
        self.full_decode = source / "historical" / "fixed_configs" / "full_shift3.decode.json"
        self.assets = args.assets / "models"
        self.model = self.assets / "TimeLens-8B"
        self.jobs = self.work / "input.jsonl"
        self.meta_path = self.work / "video.json"
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(self.work / "yolo"))

    def upstream(self, name, *argv, module_path=None):
        env = dict(os.environ)
        if module_path:
            env["PYTHONPATH"] = os.pathsep.join([str(module_path), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, *map(str, argv)], cwd=self.work, env=env, check=True)

    def import_tools(self, root):
        sys.path[:0] = [str(root / "tools"), str(root / "baseline"), str(root / "scripts" / "plan_c"), str(root), str(self.cot)]

    def extract_features(self):
        meta = probe(self.video)
        meta["fps"] = meta["fps_num"] / meta["fps_den"]
        write_json(self.meta_path, meta)
        write_json(self.jobs, {"video_id": ID, "targetRatioWH": self.request["target_ratio"], "video_path": str(self.video)})
        binding = self.work / "temporal-source.json"
        files = ("tools/infer_yesno.py", "tools/build_sft_temporal.py", "tools/decode_scores.py")
        write_json(binding, {"source_sha256": {name: digest_file(self.sym / name) for name in files}})
        self.upstream("TimeLens temporal features", "-m", "aic.inference.checked_shared_script",
                      "--kind", "temporal", "--source-dir", self.sym, "--recipe", binding, "--",
                      "--model", self.model, "--gt", self.jobs, "--out", self.work / "raw.jsonl",
                      "--save-feats", self.work / "temporal", "--batch", "8", module_path=self.cot)
        row = one_probability_row(self.work / "raw.jsonl", meta)
        if row.get("scorer", {}).get("feat_layers") != [11, 19, 27, 35] or not (self.work / "temporal" / f"{ID}.npz").is_file():
            raise ValueError("TimeLens temporal feature layers or file are missing")

    def select_frames(self):
        import numpy as np
        self.import_tools(self.sym)
        import decode_scores as decode
        import ensemble_probs as ensemble
        import reframe as rf
        import trim_frames as trim
        from aic.data.reliable_video import detector_reader

        meta = read_json(self.meta_path)
        raw = one_probability_row(self.work / "raw.jsonl", meta)
        ratio = self.request["target_ratio"]
        width, height, n, fps = (meta[k] for k in ("width", "height", "frame_count", "fps"))
        geo = rf.geometry(width, height, ratio)
        axis = geo[0] if geo else "full"
        records = []
        groups = GROUPS if axis == "full" else GROUPS[:2]
        for group in groups:
            root = self.assets / "temporal" / group
            if read_json(root / "config.json")["folds"] != 5:
                raise ValueError(f"Expected five temporal folds: {group}")
            output = self.work / f"{group}.jsonl"
            self.upstream(group, self.sym / "tools" / "train_vlm_head.py", "--predict", "--model-dir", root,
                          "--probs", self.work / "raw.jsonl", "--feats", self.work / "temporal", "--channel", "mix",
                          "--device", "cpu", "--out", output)
            record = one_probability_row(output, meta)
            if record["windows"] != raw["windows"]:
                raise ValueError(f"Temporal windows changed: {group}")
            records.append({ID: record})

        def fuse(pair):
            return ensemble.combine(pair, [1, 1])

        base = fuse(records[:2])
        rec = fuse([base, fuse(records[2:])])[ID] if axis == "full" else base[ID]
        cfg = read_json(self.sym / "tools" / "configs" / "final_AM.json")
        zero = read_json(self.sym / "tools" / "configs" / "d3_decode.json")

        def select(params):
            scores = decode.step_probs(rec, params["temp"])
            mask = decode.select_frames(decode.frame_probs(scores, rec["windows"], fps, n, params), params)
            return [{"frame": int(frame)} for frame in np.flatnonzero(mask)]

        selected = select(zero)
        if axis == "full":
            evidence = {"mean_diff": [], "dets": [], "frames": []}
            policy = trim.load_trim(dict(cfg["trim"], tail_cap=.05, head=False))
            baseline = trim.trim_row({"predictions": selected}, evidence, n, fps, policy)[0]["predictions"]
            selected = (trim.trim_row({"predictions": select(read_json(self.full_decode))}, evidence, n, fps, policy)[0]["predictions"]
                        if baseline else [])
        elif selected:
            rf.iter_frames = detector_reader(rf.iter_frames)
            frames = [p["frame"] for p in selected]
            detector = rf.Detector(str(self.assets / "detectors" / "yolo11s.pt"), "cuda:0", cfg["imgsz"])
            stride = cfg["reframe"]["stride"]
            evidence = rf.collect(str(self.video), frames, stride, detector)
            if (evidence["frames"] != rf.sample_frames(frames, stride) or (evidence["W"], evidence["H"]) != (width, height)
                    or len(evidence["dets"]) != len(evidence["frames"])):
                raise ValueError("AM detector did not cover the selected source frames")
            selected = trim.trim_row({"predictions": selected}, evidence, n, fps, trim.load_trim(cfg["trim"]))[0]["predictions"]
        write_json(self.work / "selected.json", {"axis": axis, "frames": [p["frame"] for p in selected]})

    def predict_crops(self):
        meta, selected = read_json(self.meta_path), read_json(self.work / "selected.json")
        axis, frames = selected["axis"], selected["frames"]
        ratio = self.request["target_ratio"]
        if not frames:
            write_json(self.work / "crops.json", [])
            return

        import numpy as np
        import torch
        self.import_tools(self.src)
        from highlight.geometry import center_crop, dof_range, place_1d, safe_crop_size
        from train_spatial_head import SpatialHead, load_clip, medfilt, predict

        width, height, n, fps = (meta[k] for k in ("width", "height", "frame_count", "fps"))
        tw, th = ratio
        if axis == "full" and ratio == [9, 16]:
            box = list(map(int, center_crop(width, height, tw, th)))
            write_json(self.work / "crops.json", [{"frame": frame, "bboxes": box} for frame in frames])
            return

        indices = sample_indices(n, fps, 4 if axis == "y" else 1)
        job = self.work / "spatial.jsonl"
        write_json(job, {**read_json(self.jobs), "sampling_frame_indices": indices})
        features = {}
        for res in (448, 672):
            folder = self.work / f"spatial-{res}"
            self.upstream(f"TimeLens spatial features {res}", self.src / "scripts" / "plan_c" / "research_dense_feats.py",
                          "--gt", job, "--model", self.model, "--out-dir", folder, "--annotated-frames",
                          "--no-generate", "--max-side", res, "--batch", "8")
            features[res] = folder / f"{ID}.npz"
            with np.load(features[res], allow_pickle=False) as feat:
                if (not np.array_equal(feat["frame_indices"], indices)
                        or any(feat[k].ndim != 4 or feat[k].shape[0] != len(indices) or not np.isfinite(feat[k]).all()
                               for k in ("col", "row"))):
                    raise ValueError(f"Spatial features disagree with requested source frames: {res}")
        torch.set_num_threads(4)

        def load(res, dim):
            name = "retarget_x" if dim == "x" else "gnmc_y"
            folder = self.assets / "spatial" / f"{name}_{res}"
            config = read_json(folder / "config.json")
            if config["axis"] != dim:
                raise ValueError("Spatial checkpoint axis differs from the route")
            model = SpatialHead(config["D"]).eval().requires_grad_(False)
            model.load_state_dict(torch.load(folder / "head.pt", map_location="cpu", weights_only=True))
            return model, load_clip(features[res], dim, config["layer_idx"])

        if axis == "full":
            from research_crop_scale import CropHead, encode, geo_features, integer_box
            box = np.asarray(center_crop(width, height, tw, th), np.int64)
            row = {"width": width, "height": height, "targetRatioWH": ratio}
            vectors = []
            for res in (448, 672):
                parts = []
                for dim in ("x", "y"):
                    model, inputs = load(res, dim)
                    parts.append(encode(model, *inputs)[0])
                vectors.append(np.concatenate([*parts, np.tile(geo_features(row, box), (len(indices), 1))], axis=1))
            root = self.assets / "full16x9"
            cfg = read_json(root / "head.json")
            head = CropHead(cfg["dimension"]).eval().requires_grad_(False)
            head.load_state_dict(torch.load(root / "head.pt", map_location="cpu", weights_only=True))
            if sum(p.numel() for p in head.parameters()) != cfg["parameters"]:
                raise ValueError("Unexpected number of F1 crop head parameters")
            with torch.inference_mode():
                residuals = head(torch.from_numpy(np.stack(vectors, axis=1))).numpy()
            if residuals.shape != (len(indices), 3) or not np.isfinite(residuals).all():
                raise ValueError("F1 crop head returned invalid residuals")
            times = np.asarray(indices) / fps

            def place(frame):
                residual = [np.interp(frame / fps, times, residuals[:, j]) for j in range(3)]
                return integer_box(residual, row, box).tolist()
        else:
            series = []
            for res in (448, 672):
                model, inputs = load(res, axis)
                series.append(predict(model, *inputs, "cpu"))
            if any(len(c) != len(indices) or not np.isfinite(c).all() for c in series):
                raise ValueError("Spatial head returned invalid centers")
            if axis == "x":
                x1, x2 = ([round(float(v), 4) for v in medfilt(c)] for c in series)
                centers = medfilt([round((a + b) / 2, 4) for a, b in zip(x1, x2, strict=True)])
                times = np.arange(len(indices))
            else:
                extent = int(min(width, height * tw / th)) * th / tw / height
                centers = np.clip(np.mean(series, axis=0), extent / 2, 1 - extent / 2)
                times = np.asarray(indices) / fps
            crop_w, _ = safe_crop_size(width, height, tw, th)
            travel = dof_range(width, height, tw, th)

            def place(frame):
                center = float(np.interp(frame / fps, times, centers))
                if axis == "x":
                    return list(map(int, place_1d((center * width - crop_w / 2) / travel, width, height, tw, th)))
                x, _, w = center_crop(width, height, tw, th)
                h = w * th / tw
                return [int(x), min(max(0, round(center * height - h / 2)), int(height - h)), int(w)]

        write_json(self.work / "crops.json", [{"frame": frame, "bboxes": place(frame)} for frame in frames])

    def assemble_result(self):
        predictions = read_json(self.work / "crops.json")
        if [p["frame"] for p in predictions] != read_json(self.work / "selected.json")["frames"]:
            raise ValueError("Cropping altered the selected original frames")
        result = result_for(self.request, read_json(self.meta_path), predictions)
        pending = self.work / "validated-result.json"
        write_json(pending, result)
        pending.replace(self.result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    for name in ("source-dir", "assets", "request", "result", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    getattr(Run(args), args.stage)()


if __name__ == "__main__":
    main()
