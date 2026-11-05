#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learned crop-center head on frozen VLM spatial features / 冻结大模型空间特征上的主体中心头.

Input per second (tools/spatial_feats.py): the image-token grid of one frame
reduced to a profile along the axis (col for x, row for y) at one decoder
layer, plus the zero-shot grounding box center. A small Transformer over the
cells gives a distribution over positions; its mean is the predicted subject
center, trained (L1) on the doubao trajectory center -- the contest crops sit
exactly on it. Grouped 5-fold CV over source videos; on the labeled clips the
OOF centers are scored as crop IoU against the contest crops, next to the
zero-shot box center and the current detector rule. --predict writes per-second
centers in the local_spatial.py format (center_x / center_y lists).
--loss iou (10-21, as the teammate's R43 y head, LB 42.3): 1 - IoU of the two legal crops (clamped into the
frame) + 0.2 |center error| / crop size, instead of the center L1.
--feats-flip (10-22): features of the left-right mirrored frames; the CV also writes oof_flip.jsonl (mirrored view,
mapped back: x -> 1 - x) and --predict averages both views; --flip-train adds the mirrored samples to training.
在格子分布上学豆包主体中心；分组交叉验证，按裁剪框 IoU 与零样本定位、现有规则比较。

    python tools/train_spatial_head.py --feats outputs/spf_qwen --axis x --layer-idx 1 --out-dir outputs/sph_x
    python tools/train_spatial_head.py --predict --model-dir outputs/sph_x --feats outputs/spf_test_qwen \\
        --index $T/submission_index.jsonl --out outputs/sph_test_x.jsonl
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


class SpatialHead(nn.Module):
    def __init__(self, D, d=256, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.2), nn.Linear(D, d), nn.GELU())
        self.pos = nn.Linear(1, d)
        self.box = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 1)

    def forward(self, x, pos, box):
        """x [B, C, D], pos [B, C] cell centers in 0..1, box [B, 2] (zero-shot center, valid) -> center [B]"""
        h = self.proj(x) + self.pos(pos[..., None]) + self.box(box)[:, None, :]
        p = torch.softmax(self.out(self.enc(h)).squeeze(-1), dim=-1)
        return (p * pos).sum(-1)


def load_clip(npz, axis, li):
    z = np.load(npz)
    prof = z["col" if axis == "x" else "row"][:, li].astype(np.float16)   # [S, C, D]; float16 halves RAM (1516 portrait videos at 672)
    box = z["box"]
    c = (box[:, 0] + box[:, 2]) / 2000.0 if axis == "x" else (box[:, 1] + box[:, 3]) / 2000.0
    valid = np.isfinite(c)
    zs = np.stack([np.where(valid, c, 0.5), valid.astype(np.float32)], 1).astype(np.float32)
    return prof, zs


def medfilt(x, w=1):
    return np.array([np.median(x[max(0, i - w):i + w + 1]) for i in range(len(x))])


def batches(items, bs, rng):
    idx = list(range(len(items)))
    rng.shuffle(idx)
    for a in range(0, len(idx), bs):
        yield [items[i] for i in idx[a:a + bs]]


def to_t(batch, device):
    C = max(b[0].shape[0] for b in batch)
    X = np.zeros((len(batch), C, batch[0][0].shape[1]), np.float32)
    P = np.full((len(batch), C), 0.0, np.float32)
    for i, b in enumerate(batch):
        c = b[0].shape[0]
        X[i, :c], P[i, :c] = b[0], (np.arange(c) + 0.5) / c
        if c < C:   # pad by repeating the last cell (same position) -- grids within a video share a size
            X[i, c:], P[i, c:] = b[0][-1], P[i, c - 1]
    B = np.stack([b[1] for b in batch])
    y = np.array([b[2] for b in batch], np.float32)
    a = np.array([b[3] for b in batch], np.float32)
    return tuple(torch.from_numpy(t).to(device) for t in (X, P, B, y, a))


def crop_iou(c, t, a):
    """IoU of the legal crops of size a (fraction of the axis) centered at c and t, both clamped into the frame"""
    lp = torch.minimum((c - a / 2).clamp(min=0), 1 - a)
    lt = torch.minimum((t - a / 2).clamp(min=0), 1 - a)
    d = (lp - lt).abs()
    return (a - d).clamp(min=0) / (a + d)


def train_model(items, D, args, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = SpatialHead(D).to(args.device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.05)
    for _ in range(args.epochs):
        m.train()
        for b in batches(items, args.batch, rng):
            X, P, B, y, a = to_t(b, args.device)
            c = m(X, P, B)
            if args.loss == "iou":
                loss = (1 - crop_iou(c, y, a) + 0.2 * (c - y).abs() / a).mean()
            else:
                loss = (c - y).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m


@torch.no_grad()
def predict(m, prof, zs, device):
    m.eval()
    out = []
    for a in range(0, len(prof), 256):
        b = [(prof[k], zs[k], 0.0, 0.5) for k in range(a, min(len(prof), a + 256))]
        X, P, B, _, _ = to_t(b, device)
        out.append(m(X, P, B).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run_train(args):
    from highlight.geometry import place_1d, safe_crop_size, dof_range
    from highlight_metric import derive_h, iou_xywh
    import cv2
    rows = [json.loads(l) for l in open(args.gt, encoding="utf-8")]
    data = {}
    for r in rows:
        f = os.path.join(args.feats, r["video_id"] + ".npz")
        if not os.path.exists(f):
            continue
        prof, zs = load_clip(f, args.axis, args.layer_idx)
        tl = r["teacher_signals"]["trajectory"]
        t = np.array([p["time_sec"] for p in tl])
        c = np.array([p["center"][0 if args.axis == "x" else 1] for p in tl])
        y = np.interp(np.arange(len(prof)), t, c).astype(np.float32)
        a = 0.5
        if args.loss == "iou":
            if "width" in r:   # frame size carried in the row (training on a box without the videos)
                W, H = r["width"], r["height"]
            else:
                cap = cv2.VideoCapture(r["video_path"])
                W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
            w, h = safe_crop_size(W, H, *[float(x) for x in r["targetRatioWH"]])
            a = w / W if args.axis == "x" else h / H
        data[r["video_id"]] = {"prof": prof, "zs": zs, "y": y, "a": a, "group": r["provenance"]["source_group"], "row": r}
    if args.feats_flip:   # mirrored view (spatial_feats.py --hflip): x targets mirror, y targets stay
        for v, d in data.items():
            pf, zf = load_clip(os.path.join(args.feats_flip, v + ".npz"), args.axis, args.layer_idx)
            if len(pf) == len(d["prof"]):
                d["flip"] = (pf, zf, 1 - d["y"] if args.axis == "x" else d["y"])
    vids = sorted(data)
    D = data[vids[0]]["prof"].shape[-1]
    groups = sorted({data[v]["group"] for v in vids})
    random.Random(0).shuffle(groups)
    fold = {g: i % 5 for i, g in enumerate(groups)}
    oof, oof_f = {}, {}
    flipped = lambda keep: [(data[v]["flip"][0][s], data[v]["flip"][1][s], data[v]["flip"][2][s], data[v]["a"])  # noqa: E731
                            for v in vids if keep(v) and "flip" in data[v] for s in range(len(data[v]["y"]))]
    for k in range(5):
        tr = [(data[v]["prof"][s], data[v]["zs"][s], data[v]["y"][s], data[v]["a"]) for v in vids if fold[data[v]["group"]] != k
              for s in range(len(data[v]["y"]))]
        if args.flip_train:
            tr += flipped(lambda v: fold[data[v]["group"]] != k)
        m = train_model(tr, D, args, k + 1000 * args.seed)
        for v in vids:
            if fold[data[v]["group"]] == k:
                oof[v] = predict(m, data[v]["prof"], data[v]["zs"], args.device)
                if "flip" in data[v]:
                    pf = predict(m, data[v]["flip"][0], data[v]["flip"][1], args.device)
                    oof_f[v] = 1 - pf if args.axis == "x" else pf
        print("  fold %d: %d training seconds" % (k, len(tr)), flush=True)
    mae = {"zero-shot box": [], "head": [], "head +-1 s": []}
    for v in vids:
        y = data[v]["y"]
        zb = np.where(data[v]["zs"][:, 1] > 0, data[v]["zs"][:, 0], 0.5)
        mae["zero-shot box"].append(np.mean(np.abs(zb - y)))
        mae["head"].append(np.mean(np.abs(oof[v] - y)))
        mae["head +-1 s"].append(np.mean(np.abs(medfilt(oof[v]) - y)))
    print("mean |center - doubao| (%s, %d clips): %s" % (args.axis, len(vids), ", ".join("%s %.4f" % (k, np.mean(x)) for k, x in mae.items())))
    if args.axis == "x":
        ious = {"center": [], "zero-shot box +-1 s": [], "head +-1 s": []}
        for v in vids:
            r = data[v]["row"]
            if not r.get("cropRois"):
                continue
            cap = cv2.VideoCapture(r["video_path"]); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
            tw, th = [float(x) for x in r["targetRatioWH"]]; w, _ = safe_crop_size(W, H, tw, th); R = dof_range(W, H, tw, th)
            zb = np.where(data[v]["zs"][:, 1] > 0, data[v]["zs"][:, 0], 0.5)
            series = {"center": np.full(len(zb), 0.5), "zero-shot box +-1 s": medfilt(zb), "head +-1 s": medfilt(oof[v])}
            for f, gt in r["cropRois"]:
                for name, c in series.items():
                    cx = float(np.interp(f / fps, np.arange(len(c)), c))
                    u = min(1.0, max(0.0, (cx * W - w / 2.0) / R)) if R > 0 else 0.5
                    x0, y0, w0 = place_1d(u, W, H, tw, th)
                    ious[name].append(iou_xywh([x0, y0, w0, derive_h(w0, [tw, th])], gt))
        print("crop IoU on GT frames (%d): %s" % (len(ious["center"]), ", ".join("%s %.3f" % (k, np.mean(x)) for k, x in ious.items())))
    os.makedirs(args.out_dir, exist_ok=True)
    allitems = [(data[v]["prof"][s], data[v]["zs"][s], data[v]["y"][s], data[v]["a"]) for v in vids for s in range(len(data[v]["y"]))]
    if args.flip_train:
        allitems += flipped(lambda v: True)
    m = train_model(allitems, D, args, 99 + 1000 * args.seed)
    torch.save(m.state_dict(), os.path.join(args.out_dir, "head.pt"))
    json.dump({"D": D, "axis": args.axis, "layer_idx": args.layer_idx}, open(os.path.join(args.out_dir, "config.json"), "w"))
    with open(os.path.join(args.out_dir, "oof.jsonl"), "w") as f:
        for v in vids:
            f.write(json.dumps({"video_id": v, "center_" + args.axis: [round(float(x), 4) for x in medfilt(oof[v])]}) + "\n")
    if oof_f:   # the mirrored view's OOF centers, mapped back (average with oof.jsonl = test-time flip averaging)
        with open(os.path.join(args.out_dir, "oof_flip.jsonl"), "w") as f:
            for v in vids:
                if v in oof_f:
                    f.write(json.dumps({"video_id": v, "center_" + args.axis: [round(float(x), 4) for x in medfilt(oof_f[v])]}) + "\n")
    print("-> %s" % args.out_dir)
    return 0


def run_predict(args):
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    m = SpatialHead(cfg["D"]).to(args.device)
    m.load_state_dict(torch.load(os.path.join(args.model_dir, "head.pt"), map_location=args.device))
    ids = [json.loads(l)["video_id"] for l in open(args.index, encoding="utf-8") if l.strip()]
    n = 0
    with open(args.out, "w") as f:
        for v in ids:
            p = os.path.join(args.feats, "%s.npz" % v)
            if not os.path.exists(p):
                continue
            prof, zs = load_clip(p, cfg["axis"], cfg["layer_idx"])
            c = predict(m, prof, zs, args.device)
            pf = os.path.join(args.feats_flip, "%s.npz" % v) if args.feats_flip else None
            if pf and os.path.exists(pf):   # test-time average with the mirrored view
                q = predict(m, *load_clip(pf, cfg["axis"], cfg["layer_idx"]), args.device)
                if len(q) == len(c):
                    c = (c + (1 - q if cfg["axis"] == "x" else q)) / 2
            c = medfilt(c)
            f.write(json.dumps({"video_id": str(v), "center_" + cfg["axis"]: [round(float(x), 4) for x in c]}) + "\n")
            n += 1
    print("%d videos -> %s" % (n, args.out))
    return 0


def main():
    ap = argparse.ArgumentParser(description="Learned crop-center head on VLM spatial features")
    ap.add_argument("--gt", default="data/train.with_video.jsonl")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--axis", choices=("x", "y"), default="x")
    ap.add_argument("--layer-idx", type=int, default=1, help="index into the saved layers")
    ap.add_argument("--out-dir", default="outputs/sph")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--loss", choices=("l1", "iou"), default="l1")
    ap.add_argument("--seed", type=int, default=0, help="init seed offset (same folds), for seed ensembles")
    ap.add_argument("--feats-flip", default=None, help="features of the left-right mirrored frames (spatial_feats.py --hflip)")
    ap.add_argument("--flip-train", action="store_true", help="also train on the mirrored view (needs --feats-flip)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    return run_predict(args) if args.predict else run_train(args)


if __name__ == "__main__":
    sys.exit(main())
