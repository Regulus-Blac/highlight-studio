#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild the training clips referenced by train.jsonl / 重建训练切片.

The weak-label file ships labels but no videos: `video_path` is empty on every
row and each row points at a source video through
`provenance.source_group` plus a `clip` time window. This cuts that window out
of the source and fills `video_path` in a new JSONL.
train.jsonl 只有标签没有视频，本脚本按 clip 窗口从源片切出来并回填 video_path。

    python3 tools/build_train_clips.py \
        --train train.jsonl \
        --source-dir data/video \
        --out-dir data/clips \
        --out-jsonl data/train.with_video.jsonl

Frame mapping (verified against the real files, holds at 23.98/24/25/29.97/30 fps):

    clip frame f  <->  source time  clip.start_sec + f / fps_source

so `cropRois` / `segments` frame indices are clip-relative and stay valid in the
cut file without renumbering, as long as the cut starts exactly at
round(clip.start_sec * fps) and keeps every frame after it.
帧号是 clip 相对的，只要切点对齐就无需重编号。

The script is incremental: a clip whose file already has the expected frame
count is skipped, so it can be re-run as more source videos finish downloading.
可增量重跑——已切好且帧数正确的会跳过。
"""

import os
import sys
import json
import time
import argparse
import collections

try:
    import cv2
except ImportError:
    sys.exit("[fatal] OpenCV is required: pip install opencv-python-headless")


def index_sources(source_dirs):
    """{basename-without-ext: full path} for every mp4 under the source dirs.

    Skips macOS AppleDouble files (`._x.mp4`) that exFAT drives carry.
    跳过 exFAT 盘上的 ._ 元数据文件。
    """
    have = {}
    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for fn in files:
                if fn.endswith(".mp4") and not fn.startswith("._"):
                    have.setdefault(fn[:-4], os.path.join(root, fn))
    return have


def frame_count(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def cut_clip(src_path, dst_path, start_sec, n_want, fourcc_name="mp4v"):
    """Write n_want frames starting at start_sec. Returns (n_written, fps, W, H, note)."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return 0, 0.0, 0, 0, "source unreadable"
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or W <= 0 or H <= 0:
        cap.release()
        return 0, 0.0, 0, 0, "bad source metadata"

    start_frame = int(round(start_sec * fps))
    note = ""

    # Seek, then confirm where we actually landed. Keyframe-only seeking can
    # miss; fall back to sequential decode when it does.
    # 先跳转再核对落点，跳不准就退回顺序解码。
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if landed != start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        note = "sequential seek"
        for _ in range(start_frame):
            if not cap.grab():
                cap.release()
                return 0, fps, W, H, "source ended before the clip start"

    # The temp name must keep the .mp4 extension: OpenCV picks the container
    # from the extension and refuses to open anything it does not recognise.
    # 临时文件必须保留 .mp4 后缀，OpenCV 按扩展名选容器。
    tmp_path = dst_path[:-4] + ".part.mp4"
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        return 0, fps, W, H, "VideoWriter refused to open (codec %s)" % fourcc_name

    written = 0
    while written < n_want:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()

    if written == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return 0, fps, W, H, note or "no frames decoded"

    if os.path.exists(dst_path):
        os.remove(dst_path)
    os.replace(tmp_path, dst_path)
    return written, fps, W, H, note


def main():
    ap = argparse.ArgumentParser(description="Cut training clips out of source videos")
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--source-dir", nargs="+", default=["data/video"],
                    help="one or more dirs searched recursively for <source_group>.mp4")
    ap.add_argument("--out-dir", default="data/clips")
    ap.add_argument("--out-jsonl", default="data/train.with_video.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be cut, write nothing")
    ap.add_argument("--force", action="store_true", help="recut clips that already exist")
    ap.add_argument("--fourcc", default="mp4v")
    ap.add_argument("--progress-every", type=int, default=25)
    args = ap.parse_args()

    rows = []
    with open(args.train, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]

    have = index_sources(args.source_dir)
    print("  rows          : %d" % len(rows))
    print("  source videos : %d under %s" % (len(have), ", ".join(args.source_dir)))

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        out_parent = os.path.dirname(os.path.abspath(args.out_jsonl))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)

    stats = collections.Counter()
    problems = []
    kept = []
    t0 = time.time()

    for i, d in enumerate(rows, 1):
        vid = d["video_id"]
        group = d["provenance"]["source_group"]
        src = have.get(group)
        if src is None:
            stats["no source video"] += 1
            continue

        clip = d["clip"]
        start_sec = float(clip["start_sec"])
        dur = float(clip["end_sec"]) - start_sec
        dst = os.path.join(args.out_dir, vid + ".mp4")

        # frames the labels actually reach into
        rois = d.get("cropRois") or []
        max_label_frame = max([f for f, _ in rois], default=-1)
        for s in d.get("segments") or []:
            max_label_frame = max(max_label_frame, int(s["end_frame"]))

        if args.dry_run:
            stats["would cut"] += 1
            continue

        if os.path.exists(dst) and not args.force:
            n_have = frame_count(dst)
            if n_have > max_label_frame:
                stats["skipped (already built)"] += 1
                d["video_path"] = os.path.abspath(dst)
                kept.append(d)
                continue

        cap = cv2.VideoCapture(src)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        if fps <= 0:
            stats["bad source fps"] += 1
            problems.append("%s: cannot read fps from %s" % (vid, src))
            continue
        # segments[].end_frame is inclusive and is occasionally one frame past
        # round(duration*fps), so ask for whatever the labels actually reach.
        # end_frame 是闭区间，偶尔比按时长算出的帧数多一帧，按标签实际需要取。
        n_want = max(int(round(dur * fps)), max_label_frame + 1)

        written, fps, W, H, note = cut_clip(src, dst, start_sec, n_want, args.fourcc)
        if written == 0:
            stats["cut failed"] += 1
            problems.append("%s: %s" % (vid, note or "unknown failure"))
            continue

        # The labels must fit inside what we actually wrote.
        # 标签帧号必须落在实际写出的帧数之内。
        if max_label_frame >= written:
            stats["labels overrun the clip"] += 1
            problems.append("%s: label frame %d >= written %d (wanted %d)"
                            % (vid, max_label_frame, written, n_want))
            continue
        if written < n_want:
            stats["short but usable"] += 1
        else:
            stats["cut ok"] += 1
        if note:
            stats["note: " + note] += 1

        d["video_path"] = os.path.abspath(dst)
        kept.append(d)

        if args.progress_every and i % args.progress_every == 0:
            el = time.time() - t0
            done = sum(v for k, v in stats.items() if not k.startswith("note:"))
            print("  %d/%d  %.1fs  (%.2f s/clip)"
                  % (i, len(rows), el, el / max(1, stats["cut ok"] + stats["short but usable"])))

    if not args.dry_run and kept:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for d in kept:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("=" * 72)
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("  %-28s %d" % (k, v))
    print("  %-28s %d" % ("rows written to jsonl", len(kept)))
    print("  %-28s %.1fs" % ("elapsed", time.time() - t0))
    if problems:
        print("-" * 72)
        for p in problems[:20]:
            print("  [!] %s" % p)
        if len(problems) > 20:
            print("  ... and %d more" % (len(problems) - 20))
    if not args.dry_run and kept:
        print("  jsonl -> %s" % args.out_jsonl)
        print("  clips -> %s" % args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
