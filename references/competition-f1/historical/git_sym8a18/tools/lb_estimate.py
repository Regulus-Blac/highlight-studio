#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leaderboard estimate before submitting / 提交前估计排行榜分数.

There is no labeled data that looks like the test set, so this models the
leaderboard itself. A "judge" (the strongest frame scorer we have, any size --
it only evaluates, it is never submitted) gives every test frame a highlight
probability p. Per video, the expected contest F of a submission is

    E[F] = 2 * sum_{f in sel} q_f * u_f / (|sel| + sum_{all f} q_f)
    q_f  = sigmoid(a * logit(p_f) + b + c * [new video] + d * tail(f))
    tail = clip((f / n - 0.7) / 0.3, 0, 1)   (0 before 70 % of the video, 1 at the end)
    u_f  = 1 when the submitted box is the whole frame, else u (mean IoU, --iou)

and the estimate is the mean over the scored videos x 100. `tail` carries what
the old-set trimming A/B showed (highlights rarely reach the end), which the
judge does not know. The four global numbers (a, b, c, d) are fitted to
submissions whose leaderboard score is
known (`--history`); leave-one-out refits report how far off a prediction for
an unseen submission is. Only whole-submission scores are used -- nothing is
fitted per video, and test frames are never looked at.
评判模型给每帧打分，4 个全局参数（尖锐度、整体、新视频、片尾）拟合到已知成绩；留一法报告预测误差。

    python tools/lb_estimate.py --judge outputs/ens9/test_judge.probs.jsonl \
        --history tools/configs/lb_history.json \
        --predict outputs/sub/C6T_timelens_yesdg_trimAM.jsonl
"""

import os
import sys
import json
import argparse

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_scores import DEFAULT_PARAMS, frame_probs, read_jsonl, step_probs  # noqa: E402

# Videos 0-173 are the old test set (md5-identical), 174-425 the new ones.
N_OLD = 174
# Every STRIDE-th frame stands in for its neighbours (the judge's p is linear
# between one-second step centers, so this changes the estimate by ~0.01).
STRIDE = 8
SETS = {"old174": [str(i) for i in range(N_OLD)], "semi426": [str(i) for i in range(426)]}


def tail_pos(f, n):
    return np.clip((np.asarray(f, float) / max(n, 1) - 0.7) / 0.3, 0.0, 1.0)


def load_judge(path):
    """{vid: dict(z=logit per frame, n, W, H, new)}"""
    out = {}
    for r in read_jsonl(path):
        if r.get("error") or not r.get("windows"):
            continue
        p = frame_probs(step_probs(r), r["windows"], r["fps"], r["n_frames"], DEFAULT_PARAMS)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        n = r["n_frames"]
        out[r["video_id"]] = {"z": np.log(p) - np.log1p(-p), "tail": tail_pos(np.arange(n), n), "n": n, "W": r["width"],
                              "H": r["height"], "new": int(r["video_id"]) >= N_OLD}
    return out


def load_submission(path, judge, vids):
    """Flat arrays for fast scoring: selected-frame logits, full-box flags, video index."""
    rows = {r["video_id"]: r for r in read_jsonl(path)}
    zs, tl, full, seg, nsel = [], [], [], [], np.zeros(len(vids))
    for i, vid in enumerate(vids):
        j, r = judge[vid], rows.get(vid) or {}
        tw, th = (r.get("targetRatioWH") or [16, 9])
        seen = set()
        for p in r.get("predictions") or []:
            f = int(p["frame"])
            nsel[i] += 1                      # every prediction counts in N_pred
            if f in seen or not 0 <= f < j["n"]:
                continue
            seen.add(f)
            if f % STRIDE:
                continue
            x, y, w = (float(v) for v in p["bboxes"])
            h = w * th / tw
            zs.append(j["z"][f])
            tl.append(j["tail"][f])
            full.append(w * h >= 0.97 * j["W"] * j["H"])
            seg.append(i)
    return {"z": np.array(zs), "tail": np.array(tl), "full": np.array(full, bool), "seg": np.array(seg, int),
            "nsel": nsel}


class Estimator:
    def __init__(self, judge, vids, iou):
        self.vids, self.iou = vids, iou
        self.zall = np.concatenate([judge[v]["z"][::STRIDE] for v in vids])
        self.tailall = np.concatenate([judge[v]["tail"][::STRIDE] for v in vids])
        self.segall = np.concatenate([np.full(len(judge[v]["z"][::STRIDE]), i) for i, v in enumerate(vids)])
        self.newv = np.array([judge[v]["new"] for v in vids], float)
        self._cache = (None, None)

    def unpack(self, t):
        return np.exp(t[0]), t[1], t[2], t[3], self.iou

    def expected_gt(self, t):
        """Expected N_gt per video under t (cached for the last t)."""
        key = tuple(np.round(t, 12))
        if self._cache[0] != key:
            a, b, c, d, _ = self.unpack(t)
            qall = 1 / (1 + np.exp(-(a * self.zall + b + c * self.newv[self.segall] + d * self.tailall)))
            self._cache = (key, STRIDE * np.bincount(self.segall, qall, minlength=len(self.vids)))
        return self._cache[1]

    def per_video(self, sub, t):
        a, b, c, d, u = self.unpack(t)
        k = len(self.vids)
        ngt = self.expected_gt(t)
        qs = 1 / (1 + np.exp(-(a * sub["z"] + b + c * self.newv[sub["seg"]] + d * sub["tail"])))
        s = STRIDE * np.bincount(sub["seg"], qs * np.where(sub["full"], 1.0, u), minlength=k)
        den = sub["nsel"] + ngt
        return np.where(sub["nsel"] > 0, 2 * s / np.maximum(den, 1e-9), 0.0)

    def score(self, sub, t, vset):
        idx = np.array([i for i, v in enumerate(self.vids) if v in vset])
        return self.per_video(sub, t)[idx].mean() * 100


def fit(est, items, starts=((0.0, 0.0, 0.0, 0.0), (0.5, -0.5, -0.5, -1.0), (-0.5, 0.5, 0.0, -2.0))):
    for it in items:
        it.setdefault("idx", np.array([i for i, v in enumerate(est.vids) if v in it["vset"]]))

    def loss(t):
        return sum((est.per_video(it["sub"], t)[it["idx"]].mean() * 100 - it["score"]) ** 2 for it in items)
    best = None
    for start in starts:
        r = minimize(loss, np.array(start, float), method="Nelder-Mead",
                     options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def main():
    ap = argparse.ArgumentParser(description="Leaderboard estimate calibrated on past submissions")
    ap.add_argument("--judge", required=True, help="*.probs.jsonl of the judge on all 426 test videos")
    ap.add_argument("--history", required=True, help='JSON list of {"name", "file", "score", "set": old174|semi426}')
    ap.add_argument("--predict", nargs="*", default=[], help="submission JSONL files to estimate (on semi426)")
    ap.add_argument("--iou", type=float, default=0.8, help="mean IoU of a cropped (not whole-frame) box")
    ap.add_argument("--no-loo", action="store_true")
    args = ap.parse_args()

    judge = load_judge(args.judge)
    vids = SETS["semi426"]
    missing = [v for v in vids if v not in judge]
    if missing:
        print("[fatal] judge has no record for %d videos" % len(missing))
        return 1
    est = Estimator(judge, vids, args.iou)
    base = os.path.dirname(os.path.abspath(args.history))
    items = []
    for h in json.load(open(args.history, encoding="utf-8")):
        path = h["file"] if os.path.isabs(h["file"]) else os.path.join(base, h["file"])
        items.append(dict(h, sub=load_submission(path, judge, vids), vset=set(SETS[h["set"]])))

    t = fit(est, items)
    a, b, c, d, u = est.unpack(t)
    print("fitted on %d submissions: a %.3f  b %+.3f  c(new) %+.3f  d(tail) %+.3f  [IoU(cropped) %.2f fixed]"
          % (len(items), a, b, c, d, u))
    loo = {}
    if not args.no_loo:
        for i, it in enumerate(items):
            ti = fit(est, items[:i] + items[i + 1:], starts=(tuple(t),))
            loo[it["name"]] = est.score(it["sub"], ti, it["vset"])
    print("%-14s %-8s %7s %7s %7s" % ("submission", "set", "actual", "fit", "LOO"))
    for it in items:
        fv = est.score(it["sub"], t, it["vset"])
        print("%-14s %-8s %7.2f %7.2f %7s" % (it["name"], it["set"], it["score"], fv,
                                              "%.2f" % loo[it["name"]] if loo else "-"))
    if loo:
        for s in SETS:
            e = [loo[it["name"]] - it["score"] for it in items if it["set"] == s]
            if e:
                print("LOO error %-8s: mean %+.2f, RMSE %.2f, max |err| %.2f (%d submissions)"
                      % (s, np.mean(e), np.sqrt(np.mean(np.square(e))), np.max(np.abs(e)), len(e)))
        semi = [it for it in items if it["set"] == "semi426"]
        pairs = [(x, y) for i, x in enumerate(semi) for y in semi[i + 1:] if abs(x["score"] - y["score"]) > 1e-9]
        if pairs:
            ok = sum((loo[x["name"]] - loo[y["name"]]) * (x["score"] - y["score"]) > 0 for x, y in pairs)
            print("LOO pairwise order on semi426: %d / %d correct" % (ok, len(pairs)))
    for path in args.predict:
        sub = load_submission(path, judge, vids)
        pv = est.per_video(sub, t)
        print("estimate %-44s semi426 %.2f  (old %.2f / new %.2f)"
              % (os.path.basename(path), pv.mean() * 100, pv[:N_OLD].mean() * 100, pv[N_OLD:].mean() * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
