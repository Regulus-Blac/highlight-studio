#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light temporal head on frozen features / 冻结特征上的轻量时序头.

A small Transformer over extract_feats.py's per-step features predicts how
much of each one-second step is highlight -- the fraction the VLM's digits
encode (build_sft_temporal.score_digits / 9). It trains in minutes on a CPU,
so it is the part of the temporal model a laptop without CUDA can still train.

  * loss = BCE on the soft coverage target + a within-video pairwise ranking
    term (decoding only needs the order of steps inside a video);
  * 5-fold CV on the train split, grouped by source video so clips cut from
    one source never sit on both sides of a fold; each fold early-stops on its
    held-out part by within-video AUC;
  * val / test predictions average the fold models.

Output is *.probs.jsonl with `p_sec` per step, on the same windows as
infer_scores.py, so decode_scores.py tunes and decodes it like the VLM and
ensemble_probs.py can average the two.
输出与 VLM 同格式的逐秒概率，可直接解码或与 VLM 融合。

    python3 tools/train_temporal_head.py --gt data/train.with_video.jsonl \
        --feats outputs/feats --out-dir outputs/head
    python3 tools/decode_scores.py --probs outputs/head/val.probs.jsonl --tune
    python3 tools/train_temporal_head.py --predict --model-dir outputs/head \
        --feats outputs/feats_test --out outputs/head/test.probs.jsonl
"""

import os
import sys
import json
import time
import random
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_sft_temporal import step_bounds  # noqa: E402

MAX_LEN = 64  # steps per window (windows are <= 32 s)
N_POS = 3


# ---------------------------------- data -----------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_feats(path):
    z = np.load(path)
    return z["emb"].astype(np.float32), z["low"].astype(np.float32), json.loads(str(z["meta"]))


def pos_feats(k):
    """Per step of a k-step window: relative position, index/32, log length."""
    j = np.arange(k, dtype=np.float32)
    return np.stack([(j + 0.5) / k, j / 32.0,
                     np.full(k, np.log(k + 1.0) / np.log(33.0), dtype=np.float32)], axis=1)


def coverage(windows, segments):
    """Fraction of every global step covered by `segments` (seconds)."""
    spans = [(float(s["start_sec"]), float(s["end_sec"])) for s in segments or []]
    out = []
    for s, e, k in windows:
        for j in range(k):
            a, b = step_bounds(j, k, e - s)
            a, b = s + a, s + b
            cov = sum(max(0.0, min(b, t1) - max(a, t0)) for t0, t1 in spans)
            out.append(min(1.0, cov / max(b - a, 1e-9)))
    return np.asarray(out, dtype=np.float32)


def window_slices(windows):
    out, off = [], 0
    for _, _, k in windows:
        out.append((off, off + k))
        off += k
    return out


def load_videos(gt_path, feats_dir, use_emb=True):
    """GT rows with features -> list of per-video dicts (one entry per window)."""
    vids, missing = [], 0
    for r in read_jsonl(gt_path):
        fp = os.path.join(feats_dir, r["video_id"] + ".npz")
        if not os.path.exists(fp):
            missing += 1
            continue
        emb, low, meta = load_feats(fp)
        x = np.concatenate([emb, low], axis=1) if use_emb else low
        vids.append({
            "vid": r["video_id"], "split": r.get("dataset_split") or "train",
            "group": (r.get("provenance") or {}).get("source_group") or r["video_id"],
            "x": x, "y": coverage(meta["windows"], r.get("segments")), "meta": meta,
        })
    return vids, missing


# ---------------------------------- model ----------------------------------
class Head(nn.Module):
    def __init__(self, d_in, d=128, layers=2, heads=4, drop=0.3):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in + N_POS, d), nn.GELU())
        self.pos = nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, mask):
        T = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device).clamp(max=MAX_LEN - 1))[None]
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.out(h).squeeze(-1)


def sequences(v, norm):
    """One (inputs, target) pair per window of a video, normalized."""
    mu, sd = norm
    out = []
    for a, b in window_slices(v["meta"]["windows"]):
        x = (v["x"][a:b] - mu) / sd
        out.append((np.concatenate([x, pos_feats(b - a)], axis=1), v["y"][a:b]))
    return out


def batchify(seqs, device):
    T = max(len(s[0]) for s in seqs)
    d = seqs[0][0].shape[1]
    X = np.zeros((len(seqs), T, d), dtype=np.float32)
    Y = np.zeros((len(seqs), T), dtype=np.float32)
    M = np.zeros((len(seqs), T), dtype=bool)
    for i, (x, y) in enumerate(seqs):
        X[i, :len(x)], Y[i, :len(y)], M[i, :len(x)] = x, y, True
    return (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device),
            torch.from_numpy(M).to(device))


def loss_fn(logits, y, mask, rank_w):
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")[mask].mean()
    if rank_w <= 0:
        return bce
    rank, n = logits.new_zeros(()), 0
    for b in range(logits.shape[0]):
        s, t = logits[b][mask[b]], y[b][mask[b]]
        pairs = (t[:, None] - t[None, :]) > 0.25
        if pairs.any():
            rank = rank + F.softplus(-(s[:, None] - s[None, :])[pairs]).mean()
            n += 1
    return bce + rank_w * rank / max(n, 1)


def video_auc(p, y):
    """Within-video AUC of scores p against y >= 0.5; None if one class only."""
    pos = y >= 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    _, inv = np.unique(p, return_inverse=True)
    ranks = (np.bincount(inv, weights=ranks) / np.bincount(inv))[inv]
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mean_auc(pred, vids):
    a = [video_auc(pred[v["vid"]], v["y"]) for v in vids]
    a = [x for x in a if x is not None]
    return float(np.mean(a)) if a else float("nan")


@torch.no_grad()
def predict(models, vids, norm, device):
    """{vid: per-step probability}, averaged over `models`."""
    out = {}
    for v in vids:
        seqs = sequences(v, norm)
        X, _, M = batchify(seqs, device)
        p = np.zeros(M.shape, dtype=np.float64)
        for m in models:
            m.eval()
            p += torch.sigmoid(m(X, M)).cpu().numpy()
        p /= len(models)
        out[v["vid"]] = np.concatenate([p[i, :len(s[0])] for i, s in enumerate(seqs)])
    return out


def train_fold(tr, ho, norm, d_in, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = Head(d_in, args.d, args.layers, args.heads, args.drop).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    seqs = [s for v in tr for s in sequences(v, norm)]
    best, best_auc, bad = None, -1.0, 0
    for ep in range(args.epochs):
        model.train()
        random.shuffle(seqs)
        for i in range(0, len(seqs), args.batch):
            X, Y, M = batchify(seqs[i:i + args.batch], args.device)
            loss = loss_fn(model(X, M), Y, M, args.rank_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = mean_auc(predict([model], ho, norm, args.device), ho)
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best)
    return model, best_auc, ep + 1


def group_folds(vids, k, seed):
    groups = sorted({v["group"] for v in vids})
    random.Random(seed).shuffle(groups)
    k = max(2, min(k, len(groups)))
    fold_of = {g: i % k for i, g in enumerate(groups)}
    return [[v for v in vids if fold_of[v["group"]] == i] for i in range(k)]


def to_records(pred, vids):
    for v in vids:
        m = v["meta"]
        yield {"video_id": v["vid"], "fps": m["fps"], "n_frames": m["n_frames"],
               "width": m["width"], "height": m["height"], "windows": m["windows"],
               "p_sec": [round(float(p), 5) for p in pred[v["vid"]]], "source": "head"}


def write_jsonl(path, recs):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------------------------------- main -----------------------------------
def run_train(args):
    vids, missing = load_videos(args.gt, args.feats, not args.no_emb)
    train = [v for v in vids if v["split"] != "val"]
    val = [v for v in vids if v["split"] == "val"]
    print("videos with features: train %d, val %d (no features: %d)"
          % (len(train), len(val), missing))
    if len(train) < 4:
        print("[fatal] too few training videos")
        return 1
    allx = np.concatenate([v["x"] for v in train])
    norm = (allx.mean(axis=0), allx.std(axis=0) + 1e-6)
    d_in = allx.shape[1]
    folds = group_folds(train, args.folds, args.seed)
    models, oof, t0 = [], {}, time.time()
    for i, ho in enumerate(folds):
        tr = [v for j, f in enumerate(folds) if j != i for v in f]
        model, auc, eps = train_fold(tr, ho, norm, d_in, args, args.seed + i)
        oof.update(predict([model], ho, norm, args.device))
        models.append(model)
        print("  fold %d: %d train / %d held-out videos, best held-out AUC %.3f (%d epochs, %.0fs)"
              % (i, len(tr), len(ho), auc, eps, time.time() - t0))
    print("  out-of-fold AUC on train: %.3f" % mean_auc(oof, train))
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {"d_in": d_in, "d": args.d, "layers": args.layers, "heads": args.heads,
           "drop": args.drop, "use_emb": not args.no_emb, "folds": len(models),
           "mu": norm[0].tolist(), "sd": norm[1].tolist()}
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(args.out_dir, "fold%d.pt" % i))
    write_jsonl(os.path.join(args.out_dir, "train_oof.probs.jsonl"), to_records(oof, train))
    if val:
        pv = predict(models, val, norm, args.device)
        print("  val AUC (fold ensemble): %.3f" % mean_auc(pv, val))
        n = write_jsonl(os.path.join(args.out_dir, "val.probs.jsonl"), to_records(pv, val))
        print("  %d val records -> %s" % (n, os.path.join(args.out_dir, "val.probs.jsonl")))
    print("  models -> %s" % args.out_dir)
    return 0


def run_predict(args):
    with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models = []
    for i in range(cfg["folds"]):
        m = Head(cfg["d_in"], cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"]).to(args.device)
        m.load_state_dict(torch.load(os.path.join(args.model_dir, "fold%d.pt" % i),
                                     map_location=args.device))
        models.append(m)
    norm = (np.asarray(cfg["mu"], dtype=np.float32), np.asarray(cfg["sd"], dtype=np.float32))
    vids = []
    for fn in sorted(os.listdir(args.feats)):
        if not fn.endswith(".npz"):
            continue
        emb, low, meta = load_feats(os.path.join(args.feats, fn))
        x = np.concatenate([emb, low], axis=1) if cfg["use_emb"] else low
        if x.shape[1] != cfg["d_in"]:
            print("[fatal] %s: feature dim %d != trained %d (same --backbone?)"
                  % (fn, x.shape[1], cfg["d_in"]))
            return 1
        vids.append({"vid": meta["video_id"], "x": x, "meta": meta,
                     "y": np.zeros(len(x), dtype=np.float32)})
    pred = predict(models, vids, norm, args.device)
    n = write_jsonl(args.out, to_records(pred, vids))
    print("%d records -> %s" % (n, args.out))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Light temporal head on frozen per-step features")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--feats", required=True, help="extract_feats.py output dir")
    ap.add_argument("--out-dir", default="outputs/head")
    ap.add_argument("--predict", action="store_true", help="apply saved fold models")
    ap.add_argument("--model-dir", default="outputs/head")
    ap.add_argument("--out", default=None, help="probs JSONL for --predict")
    ap.add_argument("--no-emb", action="store_true", help="cheap signals only (ablation)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--rank-w", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.predict:
        if not args.out:
            ap.error("--predict needs --out")
        return run_predict(args)
    return run_train(args)


if __name__ == "__main__":
    sys.exit(main())
