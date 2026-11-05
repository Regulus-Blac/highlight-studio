#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense highlight head on frozen VLM hidden states / 冻结 VLM 隐藏状态上的稠密打分头.

Input is what `infer_yesno.py --save-feats` writes in one pass: per video a
`<video_id>.npz` (last / vis hidden states [steps, layers, dim], REZE yes
margin [steps]) and the matching *.probs.jsonl record (window layout, fps).
A Transformer over the one-second steps of a window predicts four channels:

  cov       fraction of the step inside the teacher's segments (BCE) -> p_sec
  teacher   the Doubao-Seed teacher's per-second highlight_score (soft BCE)
  start/end segment boundary heatmaps (BCE, auxiliary)

plus the within-video pairwise ranking term on cov (train_temporal_head.loss_fn).

Augmentation aimed at the test shift (sparser highlights, longer videos):
random sub-windows; "sparse" sequences that pad a clip with contiguous
negative runs from other clips up to 32 steps and 10-20 % highlight; a few
all-negative sequences. Each step's features come from its own 3 s context,
so splicing steps from different clips is well defined.

External data (`--ext NAME:GT:PROBS:FEATS`, rows from build_ext_labels.py):
each source gets its own output channel trained on its per-second score
(soft BCE + the same ranking term); the contest channels above only see the
contest clips. The encoder is shared. Each epoch mixes `--ext-ratio` x as many
external windows as contest windows. Folds group by source_group across
sources, so a QVH full video never trains the fold that holds its contest
clip out. `--ext-heldout` lists external video ids kept out of training and
scored at the end (ext_heldout_<name>_<channel>.probs.jsonl).
外部数据各有输出通道，编码器共享；按 source_group 分折防泄漏。

5-fold CV grouped by source video on the train split, early stop on held-out
within-video AUC, fold-ensemble predictions for val / test. Outputs
*.probs.jsonl (p_sec) for decode_scores.py / ensemble_probs.py, one file per
output channel (cov / teacher / mix / start / end); boundary channels are diagnostic.
逐秒四通道：覆盖率、教师分、起止边界；稀疏拼接增强；输出与其他 probs 同格式。

    python tools/train_vlm_head.py --gt data/train.with_video.jsonl \
        --probs outputs/rz_qwen3vl8b.probs.jsonl --feats outputs/rzf_qwen3vl8b \
        --out-dir outputs/vhead_qwen3vl8b --device cuda
    python tools/decode_scores.py --probs outputs/vhead_qwen3vl8b/val_cov.probs.jsonl --tune
    python tools/train_vlm_head.py --predict --model-dir outputs/vhead_qwen3vl8b \
        --probs outputs/rz_test_qwen3vl8b.probs.jsonl --feats outputs/rzf_test_qwen3vl8b \
        --out outputs/vhead_qwen3vl8b/test.probs.jsonl
"""

import os
import sys
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_scores import step_centers  # noqa: E402
from train_temporal_head import (coverage, group_folds, loss_fn, mean_auc, pos_feats,  # noqa: E402
                                 read_jsonl, window_slices, write_jsonl)

MAX_LEN = 64      # steps per window (windows are <= 32 s)
SPARSE_LEN = 32   # synthetic sequences stay inside one inference window
N_POS = 3
N_OUT = 4         # cov, teacher, start, end
CHANNELS = ("cov", "teacher", "mix", "start", "end")


# ---------------------------------- data -----------------------------------
def load_x(path, feat, layers):
    """npz -> [steps, d] float32: chosen hidden states (+ REZE margin / 5)."""
    z = np.load(path)
    parts = []
    for name in (("vis", "last") if feat == "both" else (feat,)):
        a = z[name][:, layers, :].astype(np.float32)
        parts.append(a.reshape(a.shape[0], -1))
    parts.append((z["margin"].astype(np.float32) / 5.0)[:, None])
    return np.concatenate(parts, axis=1)


def teacher_curve(row, centers):
    """Teacher highlight_score at every step center (seconds); points with
    zero confidence (end-of-clip padding) are ignored."""
    tl = [t for t in (row.get("teacher_signals") or {}).get("timeline") or []
          if float(t.get("confidence", 1.0)) > 0]
    if len(tl) < 2:
        return None
    t = np.array([float(x["time_sec"]) for x in tl])
    s = np.array([float(x["highlight_score"]) for x in tl])
    return np.interp(centers, t, s).astype(np.float32)


def boundary_heat(segments, centers, sigma=1.0):
    st = np.zeros(len(centers), np.float32)
    en = np.zeros(len(centers), np.float32)
    c = np.asarray(centers)
    for g in segments or []:
        st = np.maximum(st, np.exp(-(c - float(g["start_sec"])) ** 2 / (2 * sigma ** 2)))
        en = np.maximum(en, np.exp(-(c - float(g["end_sec"])) ** 2 / (2 * sigma ** 2)))
    return st, en


def load_meta(probs_path):
    return {r["video_id"]: r for r in read_jsonl(probs_path) if "error" not in r}


def load_videos(gt_path, probs_path, feats_dir, feat, layers):
    meta, vids, missing = load_meta(probs_path), [], 0
    for r in read_jsonl(gt_path):
        fp = os.path.join(feats_dir, r["video_id"] + ".npz")
        m = meta.get(r["video_id"])
        if m is None or not os.path.exists(fp):
            missing += 1
            continue
        x = load_x(fp, feat, layers)
        wins = [tuple(w) for w in m["windows"]]
        centers = step_centers(wins)
        cov = coverage(wins, r.get("segments"))
        tea = teacher_curve(r, centers)
        st, en = boundary_heat(r.get("segments"), centers)
        y = np.stack([cov, cov if tea is None else tea, st, en], axis=1)
        if len(x) != len(y):
            missing += 1
            continue
        vids.append({"vid": r["video_id"], "split": r.get("dataset_split") or "train",
                     "group": (r.get("provenance") or {}).get("source_group") or r["video_id"],
                     "x": x, "y4": y, "y": cov, "meta": m})
    return vids, missing


def sequences(v, norm):
    """-> [(x, y4, src)] per inference window; src 0 = contest clip, k = k-th --ext source."""
    mu, sd = norm
    src = v.get("src", 0)
    return [((v["x"][a:b] - mu) / sd, v["y4"][a:b], src) for a, b in window_slices(v["meta"]["windows"])]


def negative_runs(seqs, min_len=2):
    """Contiguous runs of clearly negative steps (cov < 0.05, teacher < 0.5)."""
    runs = []
    for x, y, *_ in seqs:
        neg = (y[:, 0] < 0.05) & (y[:, 1] < 0.5)
        i = 0
        while i < len(neg):
            if neg[i]:
                j = i
                while j < len(neg) and neg[j]:
                    j += 1
                if j - i >= min_len:
                    runs.append((x[i:j], y[i:j]))
                i = j
            else:
                i += 1
    return runs


def take_negatives(runs, n, rng):
    xs, ys, got = [], [], 0
    while got < n:
        x, y = runs[rng.randrange(len(runs))]
        k = min(n - got, len(x))
        s = rng.randint(0, len(x) - k)
        xs.append(x[s:s + k])
        ys.append(y[s:s + k])
        got += k
    return np.concatenate(xs), np.concatenate(ys)


def augment(seq, runs, rng, args):
    x, y, src = seq
    if src:
        # external windows are long and sparse already: random crops only
        if rng.random() < args.p_crop and len(x) > 6:
            n = rng.randint(4, len(x) - 1)
            s = rng.randint(0, len(x) - n)
            return x[s:s + n], y[s:s + n], src
        return x, y, src
    x, y = _augment_contest(x, y, runs, rng, args)
    return x, y, 0


def _augment_contest(x, y, runs, rng, args):
    r = rng.random()
    if r < args.p_sparse and runs:
        n_hl = int((y[:, 0] >= 0.5).sum())
        if n_hl:
            total = min(SPARSE_LEN, max(len(x), int(np.ceil(n_hl / rng.uniform(*args.sparse_range)))))
            n_pad = total - len(x)
            if n_pad > 0:
                nx, ny = take_negatives(runs, n_pad, rng)
                k = rng.randint(0, n_pad)
                x = np.concatenate([nx[:k], x, nx[k:]])
                y = np.concatenate([ny[:k], y, ny[k:]])
        return x, y
    r -= args.p_sparse
    if r < args.p_crop and len(x) > 6:
        n = rng.randint(4, len(x) - 1)
        s = rng.randint(0, len(x) - n)
        return x[s:s + n], y[s:s + n]
    r -= args.p_crop
    if r < args.p_empty and runs:
        return take_negatives(runs, rng.randint(4, SPARSE_LEN), rng)
    return x, y


def batchify(seqs, device):
    T = max(len(s[0]) for s in seqs)
    d = seqs[0][0].shape[1] + N_POS
    X = np.zeros((len(seqs), T, d), np.float32)
    Y = np.zeros((len(seqs), T, N_OUT), np.float32)
    M = np.zeros((len(seqs), T), bool)
    S = np.zeros(len(seqs), np.int64)
    for i, (x, y, *rest) in enumerate(seqs):
        k = len(x)
        X[i, :k] = np.concatenate([x, pos_feats(k)], axis=1)
        Y[i, :k], M[i, :k] = y, True
        S[i] = rest[0] if rest else 0
    return (torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device),
            torch.from_numpy(M).to(device), torch.from_numpy(S).to(device))


# ---------------------------------- model ----------------------------------
class DenseHead(nn.Module):
    def __init__(self, d_in, d=256, layers=4, heads=4, drop=0.3, in_drop=0.5, n_out=N_OUT):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(in_drop), nn.Linear(d_in + N_POS, d), nn.GELU(),
                                 nn.Dropout(drop))
        self.pos = nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

    def forward(self, x, mask):
        T = x.shape[1]
        h = self.inp(x) + self.pos(torch.arange(T, device=x.device).clamp(max=MAX_LEN - 1))[None]
        return self.out(self.enc(h, src_key_padding_mask=~mask))


def dense_loss(out, Y, M, S, args, n_ext=0):
    loss = out.new_zeros(())
    org = S == 0
    if bool(org.any()):
        o, y, m = out[org], Y[org], M[org]
        loss = loss + loss_fn(o[..., 0], y[..., 0], m, args.rank_w)
        if args.teacher_w > 0:
            loss = loss + args.teacher_w * F.binary_cross_entropy_with_logits(
                o[..., 1], y[..., 1], reduction="none")[m].mean()
        if args.bound_w > 0:
            loss = loss + args.bound_w * F.binary_cross_entropy_with_logits(
                o[..., 2:4], y[..., 2:4], reduction="none")[m].mean()
    for i in range(n_ext):
        idx = S == i + 1
        if bool(idx.any()):
            # the source's per-second score (teacher column) on its own channel
            loss = loss + args.ext_w * loss_fn(out[idx][..., N_OUT + i], Y[idx][..., 1], M[idx], args.rank_w)
    return loss


@torch.no_grad()
def predict(models, vids, norm, device, ext_names=()):
    """{channel: {vid: per-step probability}}, averaged over `models`;
    external channels are named "ext:<name>"."""
    out = {c: {} for c in list(CHANNELS) + ["ext:" + n for n in ext_names]}
    for v in vids:
        seqs = sequences(v, norm)
        X, _, M, _ = batchify(seqs, device)
        p = None
        for m in models:
            m.eval()
            q = torch.sigmoid(m(X, M)).cpu().numpy().astype(np.float64)
            p = q if p is None else p + q
        p /= len(models)
        cat = np.concatenate([p[i, :len(s[0])] for i, s in enumerate(seqs)])
        out["cov"][v["vid"]] = cat[:, 0]
        out["teacher"][v["vid"]] = cat[:, 1]
        out["mix"][v["vid"]] = 0.5 * (cat[:, 0] + cat[:, 1])
        out["start"][v["vid"]] = cat[:, 2]
        out["end"][v["vid"]] = cat[:, 3]
        for i, n in enumerate(ext_names):
            out["ext:" + n][v["vid"]] = cat[:, N_OUT + i]
    return out


def train_fold(tr, ho, norm, d_in, args, seed, ext_tr=(), ext_names=()):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n_ext = len(ext_names)
    model = DenseHead(d_in, args.d, args.layers, args.heads, args.drop, args.in_drop,
                      N_OUT + n_ext).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    org_seqs = [s for v in tr for s in sequences(v, norm)]
    ext_seqs = [s for v in ext_tr for s in sequences(v, norm)]
    runs = negative_runs(org_seqs)
    n_draw = min(len(ext_seqs), int(round(args.ext_ratio * len(org_seqs)))) if ext_seqs else 0
    best, best_auc, bad, ep = None, -1.0, 0, 0
    for ep in range(args.epochs):
        model.train()
        if n_draw:
            seqs = org_seqs + rng.sample(ext_seqs, n_draw)
            rng.shuffle(seqs)
        else:
            rng.shuffle(org_seqs)          # in place, as before: contest-only runs reproduce exactly
            seqs = org_seqs
        for i in range(0, len(seqs), args.batch):
            batch = [augment(s, runs, rng, args) for s in seqs[i:i + args.batch]]
            X, Y, M, S = batchify(batch, args.device)
            loss = dense_loss(model(X, M), Y, M, S, args, n_ext)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = mean_auc(predict([model], ho, norm, args.device)["cov"], ho)
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best)
    return model, best_auc, ep + 1


def to_records(pred, vids):
    for v in vids:
        m = v["meta"]
        yield {"video_id": v["vid"], "targetRatioWH": m.get("targetRatioWH"), "fps": m["fps"],
               "n_frames": m["n_frames"], "width": m["width"], "height": m["height"],
               "windows": m["windows"], "p_sec": [round(float(p), 5) for p in pred[v["vid"]]],
               "source": "vhead"}


def write_channels(out_dir, prefix, pred, vids, chans=CHANNELS):
    """<prefix>_<channel>.probs.jsonl per channel; ext:<name> is written as ext_<name>."""
    for c in chans:
        write_jsonl(os.path.join(out_dir, "%s_%s.probs.jsonl" % (prefix, c.replace(":", "_"))), to_records(pred[c], vids))


# ---------------------------------- main -----------------------------------
def parse_layers(s):
    return [int(x) for x in s.split(",")]


def run_train(args):
    layers = parse_layers(args.layers_idx)
    vids, missing = load_videos(args.gt, args.probs, args.feats, args.feat, layers)
    train = [v for v in vids if v["split"] != "val"]
    val = [v for v in vids if v["split"] == "val"]
    print("videos with features: train %d, val %d (missing %d); feat %s layers %s"
          % (len(train), len(val), missing, args.feat, layers))
    if len(train) < 10:
        print("[fatal] too few training videos")
        return 1
    allx = np.concatenate([v["x"] for v in train])
    norm = (allx.mean(axis=0), allx.std(axis=0) + 1e-6)
    d_in = allx.shape[1]
    ext_names, ext_all, ext_ho = load_ext(args, layers)
    chans = list(CHANNELS) + ["ext:" + n for n in ext_names]
    if ext_all:
        # external videos from a val clip's source would leak into the val scores
        val_groups = {v["group"] for v in val}
        n0 = len(ext_all)
        ext_all = [v for v in ext_all if v["group"] not in val_groups]
        print("external videos sharing a source with the val clips, kept out of training: %d" % (n0 - len(ext_all)))
    folds = group_folds(train, args.folds, args.seed)
    models, oof, t0 = [], {c: {} for c in chans}, time.time()
    for i, ho in enumerate(folds):
        tr = [v for j, f in enumerate(folds) if j != i for v in f]
        ho_groups = {v["group"] for v in ho}
        ext_tr = [v for v in ext_all if v["group"] not in ho_groups]
        model, auc, eps = train_fold(tr, ho, norm, d_in, args, args.seed + i, ext_tr, ext_names)
        for c, d in predict([model], ho, norm, args.device, ext_names).items():
            oof[c].update(d)
        models.append(model)
        print("  fold %d: %d train / %d held-out, best held-out AUC %.3f (%d epochs, %.0fs)%s"
              % (i, len(tr), len(ho), auc, eps, time.time() - t0,
                 "; external %d (%d kept out: same source video)" % (len(ext_tr), len(ext_all) - len(ext_tr))
                 if ext_all else ""), flush=True)
    for c in chans:
        if c not in ("start", "end"):
            print("  out-of-fold AUC on train [%s]: %.3f" % (c, mean_auc(oof[c], train)))
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = {"d_in": d_in, "d": args.d, "layers": args.layers, "heads": args.heads, "drop": args.drop,
           "in_drop": args.in_drop, "feat": args.feat, "layers_idx": layers, "folds": len(models),
           "n_out": N_OUT + len(ext_names), "ext": ext_names,
           "mu": norm[0].tolist(), "sd": norm[1].tolist(), "args": vars(args)}
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    for i, m in enumerate(models):
        torch.save(m.state_dict(), os.path.join(args.out_dir, "fold%d.pt" % i))
    write_channels(args.out_dir, "train_oof", oof, train, chans)
    if val:
        pv = predict(models, val, norm, args.device, ext_names)
        for c in chans:
            if c not in ("start", "end"):
                print("  val AUC (fold ensemble) [%s]: %.3f" % (c, mean_auc(pv[c], val)))
        write_channels(args.out_dir, "val", pv, val, chans)
        all_pred = {c: {**oof[c], **pv[c]} for c in chans}
        write_channels(args.out_dir, "all", all_pred, train + val, chans)
    for name in ext_names:
        hv = [v for v in ext_ho if v["ext"] == name]
        if not hv:
            continue
        ph = predict(models, hv, norm, args.device, ext_names)
        for c in ("cov", "mix", "ext:" + name, "start", "end"):
            if c not in ("start", "end"):
                print("  external held-out %s (%d videos) AUC [%s]: %.3f" % (name, len(hv), c, mean_auc(ph[c], hv)))
            write_jsonl(os.path.join(args.out_dir, "ext_heldout_%s_%s.probs.jsonl" % (name, c.replace(":", "_"))),
                        to_records(ph[c], hv))
    print("  -> %s" % args.out_dir)
    return 0


def load_ext(args, layers):
    """--ext NAME:GT:PROBS:FEATS -> (names, training videos, held-out videos)."""
    heldout = set()
    for path in args.ext_heldout or []:
        heldout |= {l.strip() for l in open(path, encoding="utf-8") if l.strip()}
    names, train, ho = [], [], []
    for i, spec in enumerate(args.ext or []):
        name, gt, probs, feats = spec.split(":")
        vids, missing = load_videos(gt, probs, feats, args.feat, layers)
        for v in vids:
            v["src"], v["ext"] = i + 1, name
        h = [v for v in vids if v["vid"] in heldout or v["vid"].split("_", 1)[-1] in heldout]
        hs = {id(v) for v in h}
        train += [v for v in vids if id(v) not in hs]
        ho += h
        names.append(name)
        print("external %s: %d videos (%d held out, %d missing), %d steps"
              % (name, len(vids), len(h), missing, sum(len(v["x"]) for v in vids)))
    return names, train, ho


def run_predict(args):
    with open(os.path.join(args.model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    models = []
    for i in range(cfg["folds"]):
        m = DenseHead(cfg["d_in"], cfg["d"], cfg["layers"], cfg["heads"], cfg["drop"],
                      cfg["in_drop"], cfg.get("n_out", N_OUT)).to(args.device)
        m.load_state_dict(torch.load(os.path.join(args.model_dir, "fold%d.pt" % i),
                                     map_location=args.device))
        models.append(m)
    norm = (np.asarray(cfg["mu"], np.float32), np.asarray(cfg["sd"], np.float32))
    vids = []
    for vid, m in load_meta(args.probs).items():
        fp = os.path.join(args.feats, vid + ".npz")
        if not os.path.exists(fp):
            print("[warn] no features for %s" % vid)
            continue
        x = load_x(fp, cfg["feat"], cfg["layers_idx"])
        if x.shape[1] != cfg["d_in"]:
            print("[fatal] %s: feature dim %d != trained %d" % (vid, x.shape[1], cfg["d_in"]))
            return 1
        vids.append({"vid": vid, "x": x, "meta": m, "y4": np.zeros((len(x), N_OUT), np.float32)})
    pred = predict(models, vids, norm, args.device, cfg.get("ext") or ())
    channel = args.channel or "cov"
    if channel not in pred:
        print("[fatal] channel %s not in %s" % (channel, sorted(pred)))
        return 1
    n = write_jsonl(args.out, to_records(pred[channel], vids))
    print("%d records (%s) -> %s" % (n, channel, args.out))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Dense highlight head on frozen VLM hidden states")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--probs", required=True, help="infer_yesno.py records (layout / meta)")
    ap.add_argument("--feats", required=True, help="infer_yesno.py --save-feats dir")
    ap.add_argument("--out-dir", default="outputs/vhead")
    ap.add_argument("--predict", action="store_true", help="apply saved fold models")
    ap.add_argument("--model-dir", default="outputs/vhead")
    ap.add_argument("--out", default=None, help="probs JSONL for --predict")
    ap.add_argument("--channel", default=None, help="--predict output channel: cov / teacher / mix / start / end / ext:<name>")
    ap.add_argument("--feat", choices=("vis", "last", "both"), default="both")
    ap.add_argument("--layers-idx", default="2", help="indices into the saved layers")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--in-drop", type=float, default=0.5)
    ap.add_argument("--rank-w", type=float, default=0.5)
    ap.add_argument("--teacher-w", type=float, default=0.5)
    ap.add_argument("--bound-w", type=float, default=0.2)
    ap.add_argument("--p-sparse", type=float, default=0.3)
    ap.add_argument("--p-crop", type=float, default=0.2)
    ap.add_argument("--p-empty", type=float, default=0.05)
    ap.add_argument("--sparse-range", type=lambda v: tuple(float(x) for x in v.split(",")), default=(0.10, 0.20),
                    help="highlight fraction drawn for sparse sequences, 'lo,hi'")
    ap.add_argument("--ext", action="append", default=[], help="external source NAME:GT:PROBS:FEATS (repeatable)")
    ap.add_argument("--ext-heldout", action="append", default=[], help="file of external video ids kept out of training")
    ap.add_argument("--ext-ratio", type=float, default=1.0, help="external windows per contest window, per epoch")
    ap.add_argument("--ext-w", type=float, default=1.0, help="loss weight of the external channels")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.predict:
        if not args.out:
            ap.error("--predict needs --out")
        return run_predict(args)
    return run_train(args)


if __name__ == "__main__":
    sys.exit(main())
