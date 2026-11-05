#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content-aware frame trimming of an all-frames submission / 按内容去帧.

Leaderboard A/B on the 复赛 test set (tools/README.md, 排行榜记录) showed
that the organizers' highlights rarely reach the end of a video and often
start at its first frame, and that static or subject-less stretches at either
end are mostly not highlight. Starting from a submission that keeps every
frame (boxes already placed by reframe.py), this drops:

  * the tail: max(tail_base, tail evidence) of the video, at most tail_cap;
    tail evidence = the trailing run of sampled frames that are static
    (mean |frame diff| < max(0.5, static_k x the video's median)) or show no
    detection with conf >= subject_conf, as a fraction of the samples;
  * the head (head=true): the leading run of static or subject-less samples,
    if it ends within head_cap of the video.

At least min_keep_s seconds are always kept. The evidence comes from the
detection cache reframe.py already wrote (motion + YOLO boxes on every
`stride`-th frame), so trimming adds no model and no parameters.
不加模型：用 reframe.py 已缓存的运动量与检测结果，去掉片尾（比例 + 静止/无主体证据）
和片头的静止/无主体段，至少保留 min_keep_s 秒。

    python3 tools/trim_frames.py --pred outputs/submit_allframes_reframe.jsonl \
        --video-dir baseline/video --config tools/configs/final.json --out trimmed.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
import reframe as rf  # noqa: E402
from baseline_qwen import video_meta  # noqa: E402

DEFAULT_TRIM = {
    "tail_base": 0.05,      # always drop this fraction of the tail
    "tail_cap": 0.35,       # never drop more than this fraction of the tail
    "head": True,           # also drop a static / subject-less opening
    "head_cap": 0.35,       # ... only if it ends within this fraction of the video
    "static_k": 0.15,       # static: motion < max(0.5, static_k x video median)
    "subject_conf": 0.25,   # subject-less: no detection at or above this conf
    "min_keep_s": 3.0,      # never keep less than this many seconds
}


def load_trim(cfg=None):
    out = dict(DEFAULT_TRIM)
    out.update({k: v for k, v in (cfg or {}).items() if k in DEFAULT_TRIM})
    return out


def _static_threshold(md, static_k):
    return max(0.5, static_k * float(np.median(md)))


def _has_subject(D, conf):
    return any(d[1] >= conf for d in D)


def tail_evidence(cache, static_k, conf):
    """(static, subject-less) trailing-run length as a fraction of the samples.
    The first sample has no previous frame (motion 0), so it is left out."""
    md = np.asarray(cache["mean_diff"][1:], dtype=np.float64)
    dets = cache["dets"][1:]
    if len(md) < 5:
        return 0.0, 0.0
    thr = _static_threshold(md, static_k)
    k = 0
    while k < len(md) and md[-1 - k] < thr:
        k += 1
    j = 0
    while j < len(dets) and not _has_subject(dets[-1 - j], conf):
        j += 1
    return k / float(len(md)), j / float(len(md))


def tail_cut(cache, n, fps, cfg):
    """Frames to drop from the end."""
    s, ns = tail_evidence(cache, cfg["static_k"], cfg["subject_conf"])
    f = min(cfg["tail_cap"], max(cfg["tail_base"], s, ns))
    return min(int(round(f * n)), max(0, n - int(round(cfg["min_keep_s"] * fps))))


def head_cut(cache, n, cfg):
    """First frame to keep (0 = keep the opening)."""
    md = np.asarray(cache["mean_diff"], dtype=np.float64).copy()
    if len(md) > 1:
        md[0] = md[1]
    thr = _static_threshold(md[1:] if len(md) > 1 else md, cfg["static_k"])
    static = md < thr
    empty = [not _has_subject(D, cfg["subject_conf"]) for D in cache["dets"]]
    fr = cache["frames"]
    k = 0
    while k < len(fr) - 1 and (static[k] or empty[k]):
        k += 1
    if 0 < k and fr[k] <= cfg["head_cap"] * n:
        return int(fr[k])
    return 0


def trim_row(row, cache, n, fps, cfg):
    """-> (trimmed row, {"tail": frames cut, "head": first kept frame})."""
    preds = row.get("predictions") or []
    tail = tail_cut(cache, n, fps, cfg)
    keep = [p for p in preds if int(p["frame"]) < n - tail]
    head = head_cut(cache, n, cfg) if cfg["head"] else 0
    if head > 0:
        cand = [p for p in keep if int(p["frame"]) >= head]
        if len(cand) >= cfg["min_keep_s"] * fps:
            keep = cand
        else:
            head = 0
    return dict(row, predictions=keep), {"tail": tail, "head": head}


def main():
    ap = argparse.ArgumentParser(description="Content-aware frame trimming (no model)")
    ap.add_argument("--pred", required=True, help="submission whose boxes reframe.py placed")
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--config", required=True, help="JSON with 'reframe' and 'trim' sections")
    ap.add_argument("--model", default=rf.DEFAULT_MODEL, help="detector whose cache to read")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_all = json.load(f)
    params = rf.load_params(**{k: v for k, v in cfg_all["reframe"].items() if k in rf.DEFAULT_PARAMS})
    cfg = load_trim(cfg_all.get("trim"))
    tag = rf.detector_tag(args.model, args.imgsz)
    n_rows = dropped = total = missing = 0
    with open(args.pred, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            path = os.path.join(args.video_dir, str(row["video_id"]) + ".mp4")
            frames = [int(p["frame"]) for p in row.get("predictions") or []]
            cache = rf.load_cached(path, frames, params["stride"], tag, args.cache_dir) if frames else None
            if cache is None:
                missing += 1          # no evidence: keep the row as it is
                out = row
            else:
                n, fps, _, _ = video_meta(path)
                out, _ = trim_row(row, cache, n, fps, cfg)
            total += len(row.get("predictions") or [])
            dropped += len(row.get("predictions") or []) - len(out["predictions"])
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("trimmed %d rows: dropped %.1f%% of frames (%d rows without a detection cache kept)"
          % (n_rows, 100.0 * dropped / max(total, 1), missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
