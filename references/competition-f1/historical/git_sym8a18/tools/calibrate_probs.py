#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate zero-shot REZE margins into per-step probabilities / 零样本 margin 校准.

An untouched VLM answers "Yes" to almost every moment, so p = sigmoid(margin)
saturates at 1.0 and decode_scores.py can no longer rank steps. The margin
itself still ranks them. This fits a 4-parameter logistic map on the *train*
split only (the VLM stays frozen):

    p = sigmoid(w0 + w1 * m/10 + w2 * (m - mean_v)/10 + w3 * (m - mean_v)/std_v)

with m the step's yes margin and mean_v / std_v over the video's steps,
against the soft coverage target. It rewrites `p_sec` for every record
(train / val / test), so val numbers stay held-out.
只在 train 上拟合 4 个参数，VLM 不训练；val / test 只应用。

    python tools/calibrate_probs.py --probs outputs/rz_qwen3vl8b.probs.jsonl \
        --gt data/train.with_video.jsonl --out outputs/rzc_qwen3vl8b.probs.jsonl \
        --params-out outputs/rzc_qwen3vl8b.json
    python tools/calibrate_probs.py --probs outputs/rz_test_qwen3vl8b.probs.jsonl \
        --params outputs/rzc_qwen3vl8b.json --out outputs/rzc_test_qwen3vl8b.probs.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_temporal_head import coverage, read_jsonl, write_jsonl  # noqa: E402


def step_features(margin):
    m = np.asarray(margin, dtype=np.float64)
    c = m - m.mean()
    sd = m.std() if len(m) > 1 else 0.0
    return np.stack([np.ones_like(m), m / 10.0, c / 10.0, c / (sd + 1.0)], axis=1)


def fit_logistic(X, y, l2=1e-3, iters=200):
    """Newton / IRLS on soft targets; returns weights."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def main():
    ap = argparse.ArgumentParser(description="Calibrate REZE yes margins into p_sec")
    ap.add_argument("--probs", required=True, help="infer_yesno.py records (with yes_margin)")
    ap.add_argument("--gt", default=None, help="train.with_video.jsonl (fit on its train split)")
    ap.add_argument("--params", default=None, help="apply saved weights instead of fitting")
    ap.add_argument("--params-out", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [r for r in read_jsonl(args.probs) if "error" not in r and r.get("yes_margin")]
    if args.params:
        with open(args.params, "r", encoding="utf-8") as f:
            w = np.asarray(json.load(f)["w"])
    else:
        if not args.gt:
            ap.error("need --gt to fit (or --params to apply)")
        gt = {r["video_id"]: r for r in read_jsonl(args.gt)}
        X, Y = [], []
        for r in recs:
            g = gt.get(r["video_id"])
            if g is None or (g.get("dataset_split") or "train") == "val":
                continue
            X.append(step_features(r["yes_margin"]))
            Y.append(coverage([tuple(x) for x in r["windows"]], g.get("segments")))
        X, Y = np.concatenate(X), np.concatenate(Y)
        w = fit_logistic(X, Y)
        p = 1.0 / (1.0 + np.exp(-X @ w))
        print("fit on %d train steps: w = %s, mean p %.3f vs mean target %.3f"
              % (len(Y), np.round(w, 4).tolist(), p.mean(), Y.mean()))
        if args.params_out:
            with open(args.params_out, "w", encoding="utf-8") as f:
                json.dump({"w": w.tolist(), "fit_on": args.probs, "n_steps": int(len(Y))}, f, indent=2)
    out = []
    for r in recs:
        p = 1.0 / (1.0 + np.exp(-step_features(r["yes_margin"]) @ w))
        r = dict(r, p_sec=[round(float(v), 5) for v in p], source="rezec")
        out.append(r)
    n = write_jsonl(args.out, out)
    print("%d records -> %s" % (n, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
