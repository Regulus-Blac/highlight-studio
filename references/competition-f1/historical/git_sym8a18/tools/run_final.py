#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final pipeline, raw videos -> submission zip / 最终流水线（一键复现提交）.

    index + videos
      -> every frame of every video                    (no temporal model)
      -> reframe.py: subject-following crop boxes      (YOLO11s, the only model)
      -> trim_frames.py: content-aware head / tail trim (reuses the detection cache)
      -> validate_submission.py
      -> <out>.zip holding predictions.jsonl

Inference uses one model, YOLO11s (9.46M parameters), so the 复赛 parameter
factor is 1.00 (<= 100M tier). All tuned numbers live in the config
(tools/configs/final_*.json); tools/README.md 排行榜记录 has the A/B behind each.
推理只用 YOLO11s（9.46M，系数 1.00）；所有参数在 config 里，来源见 README 排行榜记录。

A video that cannot be read gets `predictions: []`; one whose reframe or trim
step fails keeps every frame with centered boxes. Either way the batch goes on.
单个视频出错不中断：读不了输出空列表；换框/去帧出错则退回全帧居中框。

    python3 tools/run_final.py --config tools/configs/final_AH.json \
        --index baseline/test_index.json --video-dir baseline/video \
        --model ~/models/yolo11s.pt --out outputs/final/final_AH
    # -> outputs/final/final_AH.jsonl, outputs/final/final_AH.zip
"""

import os
import sys
import json
import time
import zipfile
import argparse
import subprocess
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import reframe as rf  # noqa: E402
import trim_frames as tf  # noqa: E402
from baseline_qwen import video_meta  # noqa: E402


def read_index(path):
    """[(video_id, [tw, th] as ints)] from test_index.json (list) or JSONL."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        items = json.load(f) if head == "[" else [json.loads(l) for l in f if l.strip()]
    return [(str(it["video_id"]), [int(v) for v in it["targetRatioWH"]]) for it in items]


def all_frames_row(vid, ratio, n, W, H):
    """Every frame with the centered safe crop (what reframe starts from)."""
    boxes = rf.place_boxes(range(n), W, H, ratio, None)
    return {"video_id": vid, "targetRatioWH": ratio,
            "predictions": [{"frame": f, "bboxes": boxes[f]} for f in range(n)]}


def main():
    ap = argparse.ArgumentParser(description="Final pipeline: videos -> validated submission zip")
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True, help="output prefix: writes <out>.jsonl and <out>.zip")
    ap.add_argument("--model", default=None, help="YOLO weights (default: config 'detector')")
    ap.add_argument("--pose-model", default=rf.DEFAULT_POSE_MODEL, help="only if reframe.head is on")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)
    model = args.model or cfg_all.get("detector", rf.DEFAULT_MODEL)
    imgsz = int(cfg_all.get("imgsz", 640))
    params = rf.load_params(**{k: v for k, v in cfg_all["reframe"].items() if k in rf.DEFAULT_PARAMS})
    trim_cfg = tf.load_trim(cfg_all.get("trim"))
    tag = rf.detector_tag(model, imgsz)
    print("config %s\n  reframe %s\n  trim    %s" % (args.config, params, trim_cfg))

    det, pose = [], []

    def get_detector():
        if not det:
            det.append(rf.Detector(model, args.device, imgsz))
            print("detector: %s on %s" % (det[0].tag, det[0].device))
        return det[0]

    def get_pose():
        if not pose:
            pose.append(rf.PoseDetector(args.pose_model, args.device, imgsz))
        return pose[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    jsonl = args.out + ".jsonl"
    index = read_index(args.index)
    stats = collections.Counter()
    stats["_offsets"] = []
    kept = total = 0
    t0 = time.time()
    with open(jsonl, "w", encoding="utf-8") as fout:
        for i, (vid, ratio) in enumerate(index, 1):
            path = os.path.join(args.video_dir, vid + ".mp4")
            n, fps, W, H = video_meta(path) if os.path.exists(path) else (0, 0.0, 0, 0)
            if n <= 0 or W <= 0 or H <= 0:
                print("[warn] %s: unreadable video -> predictions []" % vid)
                stats["unreadable"] += 1
                row = {"video_id": vid, "targetRatioWH": ratio, "predictions": []}
            else:
                row = all_frames_row(vid, ratio, n, W, H)
                total += n
                try:
                    row = rf.reframe_row(row, path, params, get_detector, args.cache_dir, stats, get_pose)
                    cache = rf.load_cached(path, list(range(n)), params["stride"], tag, args.cache_dir)
                    if cache is None:
                        # No free axis (source ratio == target): reframe skipped
                        # detection, but the trim still needs its motion / subject
                        # evidence. 无自由轴时 reframe 不检测，这里补跑以供去帧。
                        cache = rf.collect(path, list(range(n)), params["stride"], get_detector(),
                                           args.cache_dir)
                        stats["detected for trim only"] += 1
                    row, _ = tf.trim_row(row, cache, n, fps, trim_cfg)
                except Exception as e:  # noqa: BLE001  one bad video must not stop the batch
                    print("[warn] %s: %s -> all frames, centered boxes" % (vid, e))
                    stats["fallback"] += 1
                    row = all_frames_row(vid, ratio, n, W, H)
            kept += len(row["predictions"])
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(index):
                print("  %d/%d  %.0fs" % (i, len(index), time.time() - t0), flush=True)
    stats.pop("_offsets")
    print("wrote %s: %d videos, kept %.1f%% of frames  %s" % (jsonl, len(index), 100.0 * kept / max(total, 1), dict(stats)))

    res = subprocess.run([sys.executable, os.path.join(HERE, "validate_submission.py"), "--pred", jsonl,
                          "--index", args.index, "--video-dir", args.video_dir],
                         capture_output=True, text=True)
    verdict = [l.strip() for l in res.stdout.splitlines() if l.strip() in ("PASS", "FAIL")]
    print("validate: exit %d %s" % (res.returncode, verdict[0] if verdict else ""))
    if res.returncode != 0:
        print(res.stdout[-2000:])
        return 1

    zpath = args.out + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jsonl, arcname="predictions.jsonl")
    print("zip -> %s (predictions.jsonl)" % zpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
