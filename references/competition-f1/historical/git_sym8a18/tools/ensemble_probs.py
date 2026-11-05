#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weighted average of per-step highlight probabilities / 多个逐秒概率加权平均.

Averages the expected covered fraction of every step (decode_scores.step_probs
at temp 1) across *.probs.jsonl files -- e.g. the VLM and the light temporal
head -- and writes `p_sec` records that decode_scores.py reads as is (its
`temp` then acts on the averaged logit). A video that is missing, failed, or
laid out on different windows in one input is taken from the others.

    python3 tools/ensemble_probs.py --probs outputs/scores_val_ep2.probs.jsonl \
        outputs/head/val.probs.jsonl --weights 0.6 0.4 --out outputs/ens_val.probs.jsonl
    # pick the second input's weight by val ranking AUC (two inputs)
    python3 tools/ensemble_probs.py --probs A.probs.jsonl B.probs.jsonl --sweep \
        --gt data/train.with_video.jsonl

The weight is chosen on the threshold-free ranking AUC; then run
decode_scores.py --tune on the chosen ensemble as for any single model.
权重按 val 排序 AUC 选，再对融合结果跑 decode_scores.py --tune。
"""

import sys
import json
import argparse
import collections

import numpy as np

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_scores import load_val_videos, ranking_auc, read_jsonl, step_probs  # noqa: E402


def usable(rec):
    return bool(rec) and not rec.get("error") and bool(rec.get("windows"))


def same_layout(a, b):
    wa, wb = a["windows"], b["windows"]
    return len(wa) == len(wb) and all(
        abs(float(x[0]) - float(y[0])) < 1e-3 and abs(float(x[1]) - float(y[1])) < 1e-3
        and int(x[2]) == int(y[2]) for x, y in zip(wa, wb))


def combine(inputs, weights, stats=None):
    """inputs: [{vid: rec}] -> {vid: averaged p_sec record}."""
    stats = stats if stats is not None else collections.Counter()
    order = list(dict.fromkeys(v for d in inputs for v in d))
    out = {}
    for vid in order:
        cands = [(w, d.get(vid)) for w, d in zip(weights, inputs) if usable(d.get(vid))]
        if not cands:
            out[vid] = next(d[vid] for d in inputs if vid in d)  # failed everywhere
            stats["no usable input"] += 1
            continue
        ref = cands[0][1]
        k = sum(int(w[2]) for w in ref["windows"])
        ps, ws = [], []
        for w, r in cands:
            p = step_probs(r, 1.0)
            if not same_layout(ref, r) or len(p) != k:
                stats["skipped: different windows"] += 1
                continue
            ps.append(p)
            ws.append(w)
        ws = np.asarray(ws, dtype=np.float64)
        if ws.sum() <= 0:
            ws = np.ones(len(ps))
        p = np.average(np.stack(ps), axis=0, weights=ws)
        stats["inputs used = %d" % len(ps)] += 1
        out[vid] = {"video_id": vid, "fps": ref.get("fps"), "n_frames": ref.get("n_frames"),
                    "width": ref.get("width"), "height": ref.get("height"),
                    "windows": ref["windows"], "p_sec": [round(float(v), 5) for v in p],
                    "source": "ensemble"}
    return out


def main():
    ap = argparse.ArgumentParser(description="Average per-step probabilities of several models")
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true", help="AUC per weight of input 2 (2 inputs)")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    args = ap.parse_args()

    inputs = [{r["video_id"]: r for r in read_jsonl(p)} for p in args.probs]
    if args.sweep:
        if len(inputs) != 2:
            ap.error("--sweep takes exactly two --probs")
        print("  w(input 2)   val ranking AUC")
        best = (-1.0, None)
        for w in np.linspace(0.0, 1.0, 11):
            ens = combine(inputs, [1.0 - w, w])
            auc, n = ranking_auc(load_val_videos(args.gt, ens), ens,
                                 sorted(load_val_videos(args.gt, ens)))
            print("  %.1f          %.4f  (%d videos)" % (w, auc, n))
            best = max(best, (auc, round(float(w), 1)))
        print("  best: w=%.1f (AUC %.4f) -> rerun with --weights %.1f %.1f --out ..."
              % (best[1], best[0], 1.0 - best[1], best[1]))
        return 0
    weights = args.weights or [1.0] * len(inputs)
    if len(weights) != len(inputs):
        ap.error("--weights needs one value per --probs")
    if not args.out:
        ap.error("--out is required unless --sweep")
    stats = collections.Counter()
    out = combine(inputs, weights, stats)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("%d records -> %s  %s" % (len(out), args.out, dict(stats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
