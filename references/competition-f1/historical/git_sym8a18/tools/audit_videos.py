#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video geometry audit / 视频几何审计.

Reports W, H, fps, frame count and the crop geometry implied by the target
ratio, for every video in an index. Run it the moment a new video set lands
-- it answers "how much of the spatial problem is already solved?" before any
model work. 新视频到手先跑这个，它回答"空间问题有多少是白送的"。

    # official test set
    python3 tools/audit_videos.py --index baseline/test_index.json \
        --video-dir baseline/video

    # the weak-label training clips, via their source videos
    python3 tools/audit_videos.py --train train.jsonl --source-dir data/video

Key column is `free_axis`:
  x    source is wider than the target ratio -> crop slides horizontally
  y    source is taller than the target ratio -> crop slides vertically
  none source ratio already matches the target -> the crop IS the full frame,
       so every frame scores IoU 1.0 for free and only the temporal problem
       remains. 源比例与目标一致时构图框就是整幅画面，空间部分白送。

Crop width is computed with floor(), never round(): round() can make the
evaluator's derived h = w*target_h/target_w overflow the frame by a fraction
of a pixel, which is an out-of-bounds (invalid) prediction.

`--deep` also decodes every file end to end and compares what the pipeline
assumes (CAP_PROP width/height/frame count) with what a decoder actually
delivers: the decoded frame size, the rotation tag and the true frame count.
A size mismatch puts every box of that video in the wrong frame, and an
overstated count emits frames past the end -- both score 0 for the whole
video or its tail. 深度检查：解码尺寸 / 旋转标记 / 实际帧数与 CAP_PROP 是否一致。
"""

import os
import sys
import json
import argparse
import collections

try:
    import cv2
except ImportError:
    sys.exit("[fatal] OpenCV is required: pip install opencv-python-headless")


def probe(path, deep=False):
    """-> dict with width/height/fps/n_frames, or None if unreadable.

    `deep` adds decoded_width/decoded_height (first decoded frame), rotation
    (container tag, degrees) and n_decoded (frames actually delivered).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    if deep:
        info.update(_decode_census(cap))
    cap.release()
    if info["width"] <= 0 or info["height"] <= 0:
        return None
    info["duration"] = (info["n_frames"] / info["fps"]) if info["fps"] > 0 else 0.0
    return info


def _decode_census(cap):
    """Decode the whole stream: first-frame size, rotation tag, true count."""
    rot = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0.0
    ok, frame = cap.read()
    dh, dw = frame.shape[:2] if ok else (0, 0)
    n = 1 if ok else 0
    while cap.grab():
        n += 1
    return {"decoded_width": int(dw), "decoded_height": int(dh),
            "rotation": int(round(rot or 0)) % 360, "n_decoded": n}


def deep_flags(r):
    """Problems found by --deep for one row. 深度检查发现的问题。"""
    out = []
    if (r["decoded_width"], r["decoded_height"]) != (r["width"], r["height"]):
        out.append("size %dx%d vs decoded %dx%d" % (r["width"], r["height"],
                                                     r["decoded_width"], r["decoded_height"]))
    if r["rotation"]:
        out.append("rotation %d" % r["rotation"])
    if r["n_decoded"] != r["n_frames"]:
        out.append("frames %d vs decoded %d" % (r["n_frames"], r["n_decoded"]))
    return out


def crop_geometry(W, H, tw, th):
    """Largest target-ratio rectangle inside WxH, using floor().

    Returns (cw, ch, free_axis). 返回最大内接框与自由轴。
    """
    target = float(tw) / float(th)
    src = float(W) / float(H)
    if abs(src - target) < 1e-6:
        return W, H, "none"
    if src > target:                      # source wider -> full height, slide on x
        cw = int(H * tw // th)
        return min(cw, W), H, "x"
    cw = W                                # source taller -> full width, slide on y
    ch = int(W * th // tw)
    return cw, min(ch, H), "y"


def load_index(path):
    """test_index.json (list) or a JSONL with video_id + targetRatioWH."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            items = json.load(f)
        else:
            items = [json.loads(l) for l in f if l.strip()]
    out = []
    for it in items:
        tr = it.get("targetRatioWH") or [16, 9]
        out.append((str(it["video_id"]), [float(tr[0]), float(tr[1])]))
    return out


def load_train(path, source_dir):
    """train.jsonl rows -> (video_id, ratio, source video path) triples."""
    have = {}
    for root, _, files in os.walk(source_dir):
        for fn in files:
            if fn.endswith(".mp4"):
                have[fn[:-4]] = os.path.join(root, fn)
    out, missing = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            g = d["provenance"]["source_group"]
            tr = d["targetRatioWH"]
            p = have.get(g)
            if p is None:
                missing += 1
                continue
            out.append((d["video_id"], [float(tr[0]), float(tr[1])], p))
    return out, missing


def main():
    ap = argparse.ArgumentParser(description="Audit video geometry vs target ratio")
    ap.add_argument("--index", default=None, help="test_index.json or a JSONL")
    ap.add_argument("--video-dir", default=None, help="<video_id>.mp4 lives here")
    ap.add_argument("--train", default=None, help="train.jsonl (uses source videos)")
    ap.add_argument("--source-dir", default=None, help="source video tree for --train")
    ap.add_argument("--json-out", default=None, help="write the per-video table as JSON")
    ap.add_argument("--csv-out", default=None, help="write the per-video table as CSV")
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N")
    ap.add_argument("--show", type=int, default=15, help="rows to print")
    ap.add_argument("--deep", action="store_true",
                    help="decode every file: decoded size, rotation tag, true frame count")
    args = ap.parse_args()

    # ---- resolve the work list -----------------------------------------
    if args.train:
        if not args.source_dir:
            sys.exit("[fatal] --train needs --source-dir")
        work, missing_src = load_train(args.train, args.source_dir)
        if missing_src:
            print("[warn] %d rows have no source video on disk (skipped)" % missing_src)
    elif args.index:
        if not args.video_dir:
            sys.exit("[fatal] --index needs --video-dir")
        work = [(vid, ratio, os.path.join(args.video_dir, vid + ".mp4"))
                for vid, ratio in load_index(args.index)]
    else:
        sys.exit("[fatal] pass --index or --train")

    if args.limit:
        work = work[:args.limit]

    # ---- probe ----------------------------------------------------------
    table, missing, unreadable = [], [], []
    for vid, ratio, path in work:
        if not os.path.exists(path):
            missing.append(vid)
            continue
        info = probe(path, deep=args.deep)
        if info is None:
            unreadable.append(vid)
            continue
        cw, ch, axis = crop_geometry(info["width"], info["height"], ratio[0], ratio[1])
        row = dict(info)
        row.update({
            "video_id": vid,
            "targetRatioWH": [ratio[0], ratio[1]],
            "source_ar": round(info["width"] / float(info["height"]), 4),
            "target_ar": round(ratio[0] / float(ratio[1]), 4),
            "crop_w": cw, "crop_h": ch, "free_axis": axis,
            "free_travel_px": (info["width"] - cw) if axis == "x"
                              else (info["height"] - ch) if axis == "y" else 0,
        })
        table.append(row)

    # ---- report ---------------------------------------------------------
    print("=" * 96)
    print("  videos in list : %d" % len(work))
    print("  probed ok      : %d" % len(table))
    if missing:
        print("  file missing   : %d   e.g. %s" % (len(missing), ", ".join(missing[:5])))
    if unreadable:
        print("  unreadable     : %d   e.g. %s" % (len(unreadable), ", ".join(unreadable[:5])))
    if not table:
        return 1
    print("=" * 96)

    dims = collections.Counter((r["width"], r["height"]) for r in table)
    fpss = collections.Counter(round(r["fps"], 2) for r in table)
    axes = collections.Counter(r["free_axis"] for r in table)
    ratios = collections.Counter(tuple(r["targetRatioWH"]) for r in table)

    print("  resolutions    : %s" % ", ".join(
        "%dx%d x%d" % (w, h, n) for (w, h), n in dims.most_common(6)))
    print("  fps            : %s" % ", ".join(
        "%g x%d" % (f, n) for f, n in fpss.most_common(6)))
    print("  target ratios  : %s" % ", ".join(
        "%g:%g x%d" % (a, b, n) for (a, b), n in ratios.most_common()))
    print("-" * 96)
    print("  free axis      :")
    for ax, n in axes.most_common():
        note = ("  <- crop == full frame, spatial IoU is free"
                if ax == "none" else "")
        print("      %-5s %4d  (%.1f%%)%s" % (ax, n, 100.0 * n / len(table), note))
    free = axes.get("none", 0)
    print("-" * 96)
    print("  %d of %d videos (%.1f%%) need NO spatial work at all."
          % (free, len(table), 100.0 * free / len(table)))
    total_frames = sum(r["n_frames"] for r in table)
    print("  total frames   : %d   total duration: %.1f min"
          % (total_frames, sum(r["duration"] for r in table) / 60.0))

    if args.deep:
        bad = [(r["video_id"], deep_flags(r)) for r in table]
        bad = [(vid, fl) for vid, fl in bad if fl]
        kinds = collections.Counter(f.split()[0] for _, fl in bad for f in fl)
        print("-" * 96)
        print("  deep check     : %d of %d videos flagged  %s"
              % (len(bad), len(table), dict(kinds) if kinds else ""))
        for vid, fl in bad[:20]:
            print("      %-24s %s" % (vid, "; ".join(fl)))
        if len(bad) > 20:
            print("      ... %d more" % (len(bad) - 20))
        if kinds.get("size") or kinds.get("rotation"):
            print("  !! size/rotation mismatch: boxes use CAP_PROP size; check which one"
                  " the evaluator reads before submitting these videos")
        if kinds.get("frames"):
            short = sum(1 for r in table if r["n_decoded"] < r["n_frames"])
            print("  !! %d videos decode fewer frames than CAP_PROP_FRAME_COUNT: frames past"
                  " the decoded end score 0 but count in N_pred" % short)

    if args.show:
        print("-" * 96)
        print("  %-24s %10s %7s %8s %9s %6s %8s"
              % ("video_id", "WxH", "fps", "frames", "target", "axis", "travel"))
        for r in table[:args.show]:
            print("  %-24s %10s %7.2f %8d %9s %6s %8d"
                  % (r["video_id"], "%dx%d" % (r["width"], r["height"]), r["fps"],
                     r["n_frames"],
                     "%g:%g" % (r["targetRatioWH"][0], r["targetRatioWH"][1]),
                     r["free_axis"], r["free_travel_px"]))
        if len(table) > args.show:
            print("  ... %d more" % (len(table) - args.show))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(table, f, ensure_ascii=False, indent=2)
        print("  json -> %s" % args.json_out)
    if args.csv_out:
        cols = ["video_id", "width", "height", "fps", "n_frames", "duration",
                "source_ar", "target_ar", "crop_w", "crop_h", "free_axis",
                "free_travel_px"]
        if args.deep:
            cols += ["decoded_width", "decoded_height", "rotation", "n_decoded"]
        with open(args.csv_out, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in table:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print("  csv  -> %s" % args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
