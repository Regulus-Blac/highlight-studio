#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External highlight datasets -> train.with_video.jsonl-style rows / 外部数据集转成训练行.

Each row carries what infer_yesno.py (`--gt`) and train_vlm_head.py (`--ext`)
read: video_id, video_path, targetRatioWH (placeholder), dataset_split="ext",
provenance.source / source_group, segments (seconds + frames) and a per-second
teacher_signals.timeline with highlight_score in [0, 1].

  qvh      QVHighlights official annotations (moment_detr highlight_*_release):
           segments = union of relevant windows; score = max over queries of
           the mean of the three 0-4 saliency ratings / 4 (0 outside windows).
           vid == train.jsonl provenance.source_group, so folds can group them.
  mrhisum  Mr.HiSum (most replayed, mr_hisum.h5 + metadata.csv; videos from
           ET-Instruct-164K, named <youtube_id>.mp4): score = gtscore (per
           second), segments = runs of gt_summary (the 15 % summary).
  tvsum    TVSum highlight moments as released in TimeIT
           (instruct_vhd_50_tvsum.json: timestamps + 1-5 saliency): score =
           saliency / 5 on the seconds holding a timestamp, segments = those
           seconds merged.
QVH 用官方相关窗口与显著性；Mr.HiSum 用最常重播；TVSum 用 TimeIT 的高光时刻。

    python tools/build_ext_labels.py qvh --ann ext/qvh_ann/highlight_train_release.jsonl \
        ext/qvh_ann/highlight_val_release.jsonl --video-dir data/video --out data/ext_qvh.jsonl
    python tools/build_ext_labels.py tvsum --ann ext/tvsum/instruct_vhd_50_tvsum.json \
        --video-dir ext/tvsum/videos --out data/ext_tvsum.jsonl
    python tools/build_ext_labels.py mrhisum --ann ext/mrhisum/mr_hisum.h5 --meta ext/mrhisum/metadata.csv \
        --ids ext/mrhisum/selected.txt --video-dir ext/mrhisum/videos --out data/ext_mrhisum.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np


def video_meta(path):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return n, fps, W, H


def find_videos(root):
    """{basename without .mp4: path} over every subfolder."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith(".mp4"):
                out[f[:-4]] = os.path.join(d, f)
    return out


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def top_segments(score, frac):
    """Contiguous runs of the top `frac` seconds of a per-second score."""
    if not len(score):
        return []
    k = max(1, int(round(frac * len(score))))
    thr = np.sort(score)[::-1][k - 1]
    sel = score >= thr
    spans, i = [], 0
    while i < len(sel):
        if sel[i]:
            j = i
            while j < len(sel) and sel[j]:
                j += 1
            spans.append([float(i), float(j)])
            i = j
        else:
            i += 1
    return spans


def make_row(vid, path, source, group, spans, score, dur, fps, n, extra=None):
    spans = [[max(0.0, a), min(dur, b)] for a, b in merge(spans) if min(dur, b) > max(0.0, a)]
    segs = [{"start_sec": round(a, 3), "end_sec": round(b, 3),
             "start_frame": int(round(a * fps)), "end_frame": min(n - 1, int(round(b * fps)) - 1)}
            for a, b in spans]
    tl = [{"time_sec": float(t), "highlight_score": round(float(s), 4), "confidence": 1.0}
          for t, s in enumerate(score)]
    row = {"video_id": vid, "video_path": path, "targetRatioWH": [16, 9], "dataset_split": "ext",
           "segments": segs, "teacher_signals": {"timeline": tl},
           "provenance": {"source": source, "source_group": group}, "duration": round(dur, 3)}
    if extra:
        row.update(extra)
    return row


def build_qvh(args, videos):
    ann = {}
    for p in args.ann:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    ann.setdefault(r["vid"], []).append(r)
    rows = []
    for vid, qs in sorted(ann.items()):
        path = videos.get(vid)
        if not path:
            continue
        n, fps, _, _ = video_meta(path)
        if n <= 0 or fps <= 0:
            continue
        dur = n / fps
        T = int(np.ceil(dur))
        score = np.zeros(T)
        spans = []
        for q in qs:
            spans += [[float(a), float(b)] for a, b in q["relevant_windows"]]
            for cid, sal in zip(q["relevant_clip_ids"], q["saliency_scores"]):
                s = float(np.mean(sal)) / 4.0
                for t in range(2 * int(cid), min(T, 2 * int(cid) + 2)):
                    score[t] = max(score[t], s)
        rows.append(make_row("qvhfull_" + vid, path, "qvh", vid, spans, score, dur, fps, n,
                             {"queries": [q["query"] for q in qs]}))
    return rows


def runs(mask):
    spans, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            spans.append([float(i), float(j)])
            i = j
        else:
            i += 1
    return spans


def build_tvsum(args, videos):
    import re
    rows = []
    for e in json.load(open(args.ann[0], encoding="utf-8")):
        vid = os.path.basename(e["video"])[:-4]
        path = videos.get(vid)
        if not path:
            continue
        a = e["QA"][0]["a"]
        nums = [list(map(float, re.findall(r"-?\d+(?:\.\d+)?", part)))
                for part in re.split(r"saliency scores", a, flags=re.I)]
        if len(nums) != 2 or len(nums[0]) != len(nums[1]) or not nums[0]:
            print("[warn] tvsum %s: cannot parse" % vid)
            continue
        n, fps, _, _ = video_meta(path)
        if n <= 0 or fps <= 0:
            continue
        dur = n / fps
        T = int(np.ceil(dur))
        score = np.zeros(T)
        for t, sc in zip(*nums):
            if 0 <= t < T:
                score[int(t)] = max(score[int(t)], sc / 5.0)
        rows.append(make_row("tvsum_" + vid, path, "tvsum", "tvsum_" + vid, runs(score > 0), score, dur, fps, n))
    return rows


def build_mrhisum(args, videos):
    import csv
    import h5py
    meta = {r["youtube_id"]: r for r in csv.DictReader(open(args.meta, encoding="utf-8"))}
    h5 = h5py.File(args.ann[0], "r")
    keep = None
    if args.ids:
        keep = [l.strip() for l in open(args.ids) if l.strip()]
    rows = []
    for yt in (keep if keep is not None else sorted(videos)):
        path, m = videos.get(yt), meta.get(yt)
        if not path or not m or m["video_id"] not in h5:
            continue
        n, fps, _, _ = video_meta(path)
        if n <= 0 or fps <= 0:
            continue
        dur = n / fps
        g = h5[m["video_id"]]
        T = min(int(np.ceil(dur)), len(g["gtscore"]))
        score = np.asarray(g["gtscore"][:T], float)
        summ = np.asarray(g["gt_summary"][:T]) > 0.5
        rows.append(make_row("mrhisum_" + yt, path, "mrhisum", "mrhisum_" + yt, runs(summ), score, dur, fps, n,
                             {"yt8m_labels": m.get("labels", "")}))
    return rows


def main():
    ap = argparse.ArgumentParser(description="External highlight datasets -> training rows")
    ap.add_argument("source", choices=("qvh", "tvsum", "mrhisum"))
    ap.add_argument("--ann", nargs="+", required=True,
                    help="qvh: highlight_*_release.jsonl; tvsum: instruct_vhd_50_tvsum.json; mrhisum: mr_hisum.h5")
    ap.add_argument("--meta", help="mrhisum: metadata.csv")
    ap.add_argument("--ids", help="mrhisum: youtube ids to keep (one per line), in this order")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    videos = find_videos(args.video_dir)
    rows = {"qvh": build_qvh, "tvsum": build_tvsum, "mrhisum": build_mrhisum}[args.source](args, videos)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    share = [sum(s["end_sec"] - s["start_sec"] for s in r["segments"]) / r["duration"] for r in rows]
    durs = [r["duration"] for r in rows]
    print("%s: %d rows -> %s | duration median %.1f s, total %.0f s | highlight share p10 %.2f median %.2f p90 %.2f"
          % (args.source, len(rows), args.out, float(np.median(durs)), float(np.sum(durs)),
             np.percentile(share, 10), np.median(share), np.percentile(share, 90)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
