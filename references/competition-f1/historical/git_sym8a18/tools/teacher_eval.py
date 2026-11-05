#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How close is a teacher's per-second timeline to the contest (doubao) labels? / 教师打分与赛方标签有多接近.

For each teacher JSONL (api_teacher.py) and the reference probs (our dense
heads), on the clips every compared source covers:
  spearman   mean per-clip rank correlation with the doubao highlight_score timeline
  AUC        within-video ranking AUC against the contest highlight segments
  tF         temporal F with the expected-F decoder, 2-fold CV of its params
Teacher scores (one per second, t = 0, 1, ...) are interpolated to the step
centers of the reference record's window layout, so every source is decoded
the same way.
在同一批片段上比较：与豆包逐秒分的秩相关、视频内 AUC、两折 CV 的 temporal F。

    python tools/teacher_eval.py --gt data/train.with_video.jsonl --split val \\
        --ref C4=outputs/ens9/T_L12_mix.probs.jsonl \\
        --teacher dsv4vis=outputs/teacher/dsv4vis_val.jsonl qwen38f=outputs/teacher/qwen38flash_val.jsonl
"""

import os
import sys
import json
import random
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_scores as ds  # noqa: E402


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def teacher_records(path, layout):
    """teacher JSONL -> probs records on the reference window layout."""
    out = {}
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)
        if "error" in t or t["video_id"] not in layout:
            continue
        base = layout[t["video_id"]]
        s = np.asarray(t["scores"], float)
        centers = ds.step_centers([tuple(w) for w in base["windows"]])
        p = np.interp(centers, np.arange(len(s)), s)
        out[t["video_id"]] = dict(base, p_sec=[round(float(v), 4) for v in np.clip(p, 1e-3, 1 - 1e-3)])
    return out


def cv_tf(videos, recs, vids, seed=0):
    ids = list(vids)
    random.Random(seed).shuffle(ids)
    folds = [sorted(ids[i::2]) for i in range(2)]
    held, cache = {}, {}
    for i in range(2):
        p, _ = ds.tune(videos, recs, folds[1 - i], cache)
        held.update(ds.evaluate(videos, recs, p, folds[i], cache))
    return float(np.mean([held[v][0] for v in vids])) * 100, float(np.mean([held[v][2] for v in vids]))


def main():
    ap = argparse.ArgumentParser(description="Teacher timelines vs the contest labels")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--split", default="val", help="val / train / all")
    ap.add_argument("--ref", nargs="+", required=True, help="NAME=probs.jsonl (our heads; also gives the window layout)")
    ap.add_argument("--teacher", nargs="*", default=[], help="NAME=api_teacher.jsonl")
    args = ap.parse_args()

    refs = {}
    for spec in args.ref:
        name, path = spec.split("=", 1)
        refs[name] = {r["video_id"]: r for r in ds.read_jsonl(path)}
    layout = next(iter(refs.values()))
    srcs = dict(refs)
    for spec in args.teacher:
        name, path = spec.split("=", 1)
        srcs[name] = teacher_records(path, layout)
    rows = {json.loads(l)["video_id"]: json.loads(l) for l in open(args.gt, encoding="utf-8")}
    videos = ds.load_val_videos(args.gt, layout, split=args.split)
    common = sorted(set(videos).intersection(*[set(s) for s in srcs.values()]))
    print("clips compared: %d (split %s)" % (len(common), args.split))
    print("%-12s %9s %7s %8s %6s" % ("source", "spearman", "AUC", "CV tF", "sel"))
    for name, recs in srcs.items():
        sp = []
        for vid in common:
            tl = [t for t in rows[vid]["teacher_signals"]["timeline"] if t.get("confidence", 1) > 0]
            centers = ds.step_centers([tuple(w) for w in recs[vid]["windows"]])
            d = np.interp(centers, [t["time_sec"] for t in tl], [t["highlight_score"] for t in tl])
            r = spearman(ds.step_probs(recs[vid]), d)
            if r is not None:
                sp.append(r)
        auc, _ = ds.ranking_auc(videos, recs, common)
        tf, sel = cv_tf(videos, recs, common)
        print("%-12s %9.3f %7.3f %8.2f %6.2f" % (name, float(np.mean(sp)), auc, tf, sel))
    full = {v: ds.video_scores(videos[v], np.ones(videos[v]["n"], bool))[0] for v in common}
    print("%-12s %9s %7s %8.2f %6.2f" % ("full video", "-", "-", float(np.mean(list(full.values()))) * 100, 1.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
