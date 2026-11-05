#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empty-list rule for videos that look like they hold no highlight / 疑似无高光视频交空列表.

The contest scores a video with no ground-truth highlight 1 only when its
prediction list is empty (any frame -> 0), while emptying a video that does
have highlights loses its whole F. This ranks the videos of a submission by
how little they look like a highlight, from automated outputs only:

  head_p90 / head_mean   dense head per-second probability (90th pct, mean)
  reze_p90               zero-shot Yes-No margin of the frozen VLM (90th pct)
  rating_p90             0-9 prompt expected rating (90th pct)
  subj_frac              share of detector samples with a subject (conf >= 0.25)
  motion_med             median frame difference of the detector samples

Each signal is ranked across the videos (low = less highlight-like) and the
ranks are averaged; the `--k` lowest videos get `predictions: []`, everything
else is copied unchanged. One global rule, no per-video choices. Percentiles
against the labeled clips (which all hold highlights) are printed for the
model signals as a reference.
按自动化输出排序，全局只清空最不像有高光的 K 个视频；其余原样保留。

    python tools/empty_rule.py --pred outputs/sub/C4T_timelens8b_L12_mix_trimAM.jsonl \
        --cache-pred outputs/sub/C4_timelens8b_L12_mix.jsonl \
        --head outputs/ens9/test_T_L12_mix.probs.jsonl --reze outputs/rz_test_timelens8b.probs.jsonl \
        --rating outputs/rz_test_timelens8b_dg.probs.jsonl --video-dir $T/video --config tools/configs/final_AM.json \
        --k 20 --out outputs/sub/E20_C4T_empty20.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reframe as rf  # noqa: E402

SIGNALS = ("head_p90", "head_mean", "reze_p90", "rating_p90", "subj_frac", "motion_med")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def per_video(path, key):
    return {r["video_id"]: np.asarray(r[key], float) for r in read_jsonl(path) if not r.get("error") and r.get(key)}


def model_signals(head, reze, rating):
    out = {}
    for vid, p in head.items():
        s = {"head_p90": np.percentile(p, 90), "head_mean": float(p.mean())}
        if vid in reze:
            s["reze_p90"] = np.percentile(reze[vid], 90)
        if vid in rating:
            s["rating_p90"] = np.percentile(rating[vid], 90)
        out[vid] = s
    return out


def visual_signals(rows, video_dir, cfg_path, model, imgsz, cache_dir, conf=0.25):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    params = rf.load_params(**{k: v for k, v in cfg["reframe"].items() if k in rf.DEFAULT_PARAMS})
    tag = rf.detector_tag(model, imgsz)
    out = {}
    for row in rows:
        frames = [int(p["frame"]) for p in row.get("predictions") or []]
        if not frames:
            continue
        path = os.path.join(video_dir, str(row["video_id"]) + ".mp4")
        c = rf.load_cached(path, frames, params["stride"], tag, cache_dir)
        if c is None or not c.get("dets"):
            continue
        md = np.asarray(c["mean_diff"][1:] or [0.0], float)
        out[row["video_id"]] = {"subj_frac": float(np.mean([any(d[1] >= conf for d in D) for D in c["dets"]])),
                                "motion_med": float(np.median(md))}
    return out


def main():
    ap = argparse.ArgumentParser(description="Empty the K least highlight-like videos of a submission")
    ap.add_argument("--pred", required=True, help="submission to copy (e.g. C4T)")
    ap.add_argument("--cache-pred", help="submission whose frames keyed the detection cache (pre-trim)")
    ap.add_argument("--head", required=True, help="dense head probs on the test set (p_sec)")
    ap.add_argument("--reze", help="infer_yesno yes/no records on the test set (yes_margin)")
    ap.add_argument("--rating", help="infer_yesno --mode digits records on the test set (p_sec)")
    ap.add_argument("--ref-head", help="head probs on the labeled clips (reference percentiles)")
    ap.add_argument("--ref-reze")
    ap.add_argument("--ref-rating")
    ap.add_argument("--video-dir")
    ap.add_argument("--config", help="JSON with a 'reframe' section (detector stride)")
    ap.add_argument("--model", default=rf.DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--cache-dir", default="outputs/det_cache")
    ap.add_argument("--k", type=int, required=True, help="number of videos to empty")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.pred)
    sig = model_signals(per_video(args.head, "p_sec"),
                        per_video(args.reze, "yes_margin") if args.reze else {},
                        per_video(args.rating, "p_sec") if args.rating else {})
    if args.video_dir and args.config:
        vis = visual_signals(read_jsonl(args.cache_pred or args.pred), args.video_dir, args.config,
                             args.model, args.imgsz, args.cache_dir)
        for vid, s in vis.items():
            sig.setdefault(vid, {}).update(s)
    vids = [r["video_id"] for r in rows if r["video_id"] in sig]
    used = [s for s in SIGNALS if sum(s in sig[v] for v in vids) >= 0.9 * len(vids)]
    ranks = np.zeros(len(vids))
    for s in used:
        x = np.array([sig[v].get(s, np.nan) for v in vids], float)
        x[np.isnan(x)] = np.nanmedian(x)
        ranks += np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(vids) - 1, 1)
    score = ranks / max(len(used), 1)
    order = np.argsort(score, kind="stable")
    chosen = {vids[i] for i in order[:args.k]}

    ref = {}
    for name, path, key, agg in (("head_p90", args.ref_head, "p_sec", 90), ("reze_p90", args.ref_reze, "yes_margin", 90),
                                 ("rating_p90", args.ref_rating, "p_sec", 90)):
        if path:
            ref[name] = np.array([np.percentile(v, agg) for v in per_video(path, key).values()])
    print("signals used: %s; emptying %d of %d videos" % (", ".join(used), len(chosen), len(rows)))
    print("%-6s %-5s %6s  %s" % ("video", "set", "score", "  ".join("%s%s" % (s, " (pct vs labeled)" if s in ref else "")
                                                                 for s in used)))
    for i in order[:args.k]:
        v = vids[i]
        cells = []
        for s in used:
            x = sig[v].get(s, float("nan"))
            pct = " (%3.0f%%)" % (100 * np.mean(ref[s] < x)) if s in ref else ""
            cells.append("%.3f%s" % (x, pct))
        print("%-6s %-5s %6.3f  %s" % (v, "new" if int(v) >= 174 else "old", score[i], "  ".join(cells)))
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            if r["video_id"] in chosen:
                r = dict(r, predictions=[])
                n += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("-> %s (%d rows emptied, %d new / %d old)" % (args.out, n, sum(int(v) >= 174 for v in chosen),
                                                       sum(int(v) < 174 for v in chosen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
