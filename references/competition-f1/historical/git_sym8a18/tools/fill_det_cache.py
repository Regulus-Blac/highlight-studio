#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill the detection cache for submission rows reframe.py skipped / 为跳过的视频补检测缓存.

reframe.py does not detect on videos with no free axis (source ratio ==
target ratio: the box is the whole frame), so trim_frames.py finds no
motion / subject evidence for them and keeps them untouched. This runs the
same detector over each such row's selected frames (same stride, same cache
key as reframe.py / trim_frames.py), as run_final.py does inline.
无自由轴的视频 reframe 不检测；这里按同一缓存键补跑，供 trim_frames.py 使用。

    python tools/fill_det_cache.py --pred outputs/sub/C1.jsonl --video-dir $T/video \
        --config tools/configs/final_AM.json --model yolo11s.pt
"""

import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reframe as rf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Detection cache for rows reframe.py skipped")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--config", required=True, help="JSON with a 'reframe' section (stride)")
    ap.add_argument("--model", default=rf.DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    params = rf.load_params(**{k: v for k, v in cfg["reframe"].items() if k in rf.DEFAULT_PARAMS})
    tag = rf.detector_tag(args.model, args.imgsz)
    det, filled, have, t0 = None, 0, 0, time.time()
    with open(args.pred, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for i, row in enumerate(rows, 1):
        frames = [int(p["frame"]) for p in row.get("predictions") or []]
        if not frames:
            continue
        path = os.path.join(args.video_dir, str(row["video_id"]) + ".mp4")
        if rf.load_cached(path, frames, params["stride"], tag, args.cache_dir) is not None:
            have += 1
            continue
        if det is None:
            det = rf.Detector(args.model, args.device, args.imgsz)
            print("detector: %s on %s" % (det.tag, det.device))
        try:
            rf.collect(path, frames, params["stride"], det, args.cache_dir)
            filled += 1
        except Exception as e:  # noqa: BLE001  one bad video must not stop the batch
            print("[warn] %s: %s" % (row["video_id"], e))
        if filled and filled % 20 == 0:
            print("  filled %d (row %d/%d, %.0fs)" % (filled, i, len(rows), time.time() - t0), flush=True)
    print("rows with cache already: %d, filled now: %d (%.0fs)" % (have, filled, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
