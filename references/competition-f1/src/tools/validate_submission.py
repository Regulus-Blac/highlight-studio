#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submission format gate / 提交格式校验器.

Run this before EVERY submission. It enforces the hard constraints in
task.md section 4.3 / 6.3, any of which can invalidate a prediction or the
whole file. 每次提交前必跑。

    python3 tools/validate_submission.py \
        --pred outputs/predictions.jsonl \
        --index baseline/test_index.json \
        --video-dir baseline/video

Checks (errors fail the run, warnings do not):
  E  malformed JSONL / line is not an object
  E  missing or empty video_id; duplicate video_id line
  E  video_id not in the index; index video_id missing from the file
  E  targetRatioWH altered from the index value
  E  predictions is not a list
  E  prediction missing frame / bboxes
  E  frame not a non-negative integer; frame >= n_frames
  E  bboxes not a 3-number triplet; NaN / Inf; w <= 0; x < 0; y < 0
  E  x+w > width, or y+h > height after h = w*target_h/target_w
  E  duplicate frame within one video
  W  frames not in ascending order
  W  video with an empty predictions list (legal, but scores 0 if it has GT)
  W  video predicting (nearly) every frame -- precision will collapse

Without --video-dir / --meta the bound checks cannot run; the report says so
and exits non-zero unless --allow-unchecked is passed.
"""

import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from highlight_metric import derive_h  # noqa: E402

MAX_SHOWN = 25


class Report(object):
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def ok(self):
        return not self.errors


def _finite_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def load_index(path):
    """test_index.json (a JSON list) or the 复赛 JSONL index (one object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        items = json.load(f) if head == "[" else [json.loads(l) for l in f if l.strip()]
    out = {}
    for it in items:
        tr = it.get("targetRatioWH", [16, 9])
        out[str(it["video_id"])] = [float(tr[0]), float(tr[1])]
    return out


def load_metas(video_dir, meta_path, video_ids):
    metas = {}
    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            for vid, m in json.load(f).items():
                metas[str(vid)] = m
    if video_dir:
        try:
            import cv2
        except ImportError:
            print("[warn] OpenCV missing; cannot read video metadata")
            return metas
        for vid in video_ids:
            p = os.path.join(video_dir, str(vid) + ".mp4")
            if not os.path.exists(p):
                continue
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                continue
            metas[str(vid)] = {
                "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            cap.release()
    return metas


def validate_video(vid, preds, ratio, meta, rep):
    """Validate one video's predictions in place, appending to `rep`."""
    if not isinstance(preds, list):
        rep.error("%s: predictions is not a list" % vid)
        return
    if not preds:
        rep.warn("%s: empty predictions (legal; scores 0 if this video has GT)" % vid)
        return

    seen = set()
    last = None
    unordered = False
    n_frames = (meta or {}).get("n_frames")
    W = (meta or {}).get("width")
    H = (meta or {}).get("height")

    for i, p in enumerate(preds):
        tag = "%s[%d]" % (vid, i)
        if not isinstance(p, dict):
            rep.error("%s: prediction is not an object" % tag)
            continue
        if "frame" not in p:
            rep.error("%s: missing field 'frame'" % tag)
            continue
        if "bboxes" not in p:
            rep.error("%s: missing field 'bboxes'" % tag)
            continue

        f = p["frame"]
        if not _finite_num(f) or float(f) != int(f):
            rep.error("%s: frame %r is not a finite integer" % (tag, f))
            continue
        f = int(f)
        if f < 0:
            rep.error("%s: frame %d < 0" % (tag, f))
            continue
        if n_frames and f >= int(n_frames):
            rep.error("%s: frame %d >= n_frames %d" % (tag, f, int(n_frames)))
            continue
        if f in seen:
            rep.error("%s: duplicate frame %d in the same video" % (tag, f))
            continue
        seen.add(f)
        if last is not None and f < last:
            unordered = True
        last = f

        bb = p["bboxes"]
        if not isinstance(bb, (list, tuple)):
            rep.error("%s: bboxes is not a list" % tag)
            continue
        if len(bb) != 3:
            rep.error("%s: bboxes has %d values, must be a triplet [x, y, w] "
                      "(do NOT submit h)" % (tag, len(bb)))
            continue
        if not all(_finite_num(v) for v in bb):
            rep.error("%s: bboxes %r contains NaN / Inf / non-numeric" % (tag, list(bb)))
            continue

        x, y, w = float(bb[0]), float(bb[1]), float(bb[2])
        if w <= 0:
            rep.error("%s: w = %g, must be > 0" % (tag, w))
            continue
        if x < 0 or y < 0:
            rep.error("%s: x=%g y=%g, both must be >= 0" % (tag, x, y))
            continue

        h = derive_h(w, ratio)
        if not math.isfinite(h) or h <= 0:
            rep.error("%s: derived h is invalid (w=%g ratio=%s)" % (tag, w, ratio))
            continue
        if W and x + w > float(W) + 1e-6:
            rep.error("%s: x+w = %g exceeds width %s" % (tag, x + w, W))
        if H and y + h > float(H) + 1e-6:
            # The classic trap: w rounded UP to the max inscribed size makes the
            # derived h overflow by a fraction of a pixel. Use floor, not round.
            # 常见陷阱：w 向上取整导致反算出的 h 超出画面，应用 floor 而非 round。
            rep.error("%s: y+h = %g exceeds height %s (h = w*%g/%g = %g); "
                      "floor() the crop width instead of round()"
                      % (tag, y + h, H, ratio[1], ratio[0], h))

    if unordered:
        rep.warn("%s: frames are not in ascending order (legal but discouraged)" % vid)
    if n_frames and len(seen) >= int(n_frames) * 0.95:
        rep.warn("%s: predicts %d of %d frames -- near-full coverage collapses "
                 "precision" % (vid, len(seen), int(n_frames)))


def main():
    ap = argparse.ArgumentParser(description="Validate a contest submission JSONL")
    ap.add_argument("--pred", required=True, help="submission JSONL")
    ap.add_argument("--index", required=True, help="official test_index.json")
    ap.add_argument("--video-dir", default=None, help="source videos (bound checks)")
    ap.add_argument("--meta", default=None,
                    help="JSON {video_id: {n_frames, width, height}}")
    ap.add_argument("--allow-unchecked", action="store_true",
                    help="pass even when video metadata is unavailable")
    args = ap.parse_args()

    rep = Report()
    index = load_index(args.index)

    # ---- parse the file -------------------------------------------------
    records = {}
    order = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                rep.error("line %d: invalid JSON: %s" % (lineno, e))
                continue
            if not isinstance(d, dict):
                rep.error("line %d: line is not a JSON object" % lineno)
                continue
            vid = str(d.get("video_id", "")).strip()
            if not vid:
                rep.error("line %d: missing or empty video_id" % lineno)
                continue
            if vid in records:
                rep.error("line %d: duplicate video_id %s" % (lineno, vid))
                continue
            records[vid] = d
            order.append(vid)

    # ---- coverage -------------------------------------------------------
    missing = [v for v in index if v not in records]
    extra = [v for v in records if v not in index]
    for v in missing[:MAX_SHOWN]:
        rep.error("index video_id %s has no line in the submission" % v)
    if len(missing) > MAX_SHOWN:
        rep.error("... and %d more index video_id with no line" % (len(missing) - MAX_SHOWN))
    for v in extra[:MAX_SHOWN]:
        rep.error("submission video_id %s is not in the index" % v)
    if len(extra) > MAX_SHOWN:
        rep.error("... and %d more unknown video_id" % (len(extra) - MAX_SHOWN))

    # ---- metadata -------------------------------------------------------
    metas = load_metas(args.video_dir, args.meta, list(index.keys()))
    no_meta = [v for v in index if v not in metas]
    if no_meta:
        msg = ("no video metadata for %d/%d video_id: frame-bound and "
               "box-bound checks could not run" % (len(no_meta), len(index)))
        if args.allow_unchecked:
            rep.warn(msg)
        else:
            rep.error(msg + " (pass --video-dir/--meta, or --allow-unchecked)")

    # ---- per video ------------------------------------------------------
    for vid in order:
        d = records[vid]
        official = index.get(vid)
        if official is None:
            continue
        if "targetRatioWH" in d:
            tr = d["targetRatioWH"]
            bad = (not isinstance(tr, (list, tuple)) or len(tr) != 2
                   or not all(_finite_num(v) for v in tr)
                   or [float(tr[0]), float(tr[1])] != official)
            if bad:
                rep.error("%s: targetRatioWH %r differs from the index value %s "
                          "-- it must be carried back unchanged" % (vid, tr, official))
        validate_video(vid, d.get("predictions", []), official, metas.get(vid), rep)

    # ---- report ---------------------------------------------------------
    n_preds = sum(len(d.get("predictions", []) or []) for d in records.values())
    print("=" * 64)
    print("  %s" % ("PASS" if rep.ok() else "FAIL"))
    print("=" * 64)
    print("  file          : %s" % args.pred)
    print("  index videos  : %d" % len(index))
    print("  lines parsed  : %d" % len(records))
    print("  predictions   : %d" % n_preds)
    print("  errors        : %d" % len(rep.errors))
    print("  warnings      : %d" % len(rep.warnings))
    if rep.errors:
        print("-" * 64)
        for m in rep.errors[:MAX_SHOWN]:
            print("  [E] %s" % m)
        if len(rep.errors) > MAX_SHOWN:
            print("  ... and %d more errors" % (len(rep.errors) - MAX_SHOWN))
    if rep.warnings:
        print("-" * 64)
        for m in rep.warnings[:MAX_SHOWN]:
            print("  [W] %s" % m)
        if len(rep.warnings) > MAX_SHOWN:
            print("  ... and %d more warnings" % (len(rep.warnings) - MAX_SHOWN))

    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
