#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-step features for the light temporal head / 轻量时序头的逐秒特征.

One feature row per one-second step, laid out exactly like infer_scores.py
(decode_scores.make_windows + build_sft_temporal.step_bounds), so the head's
per-step output drops into decode_scores.py and ensemble_probs.py unchanged.
Two frames per step (1/4 and 3/4 into it) go through:

  * a frozen image encoder -> mean embedding per step
      clip    open_clip ViT-B-32 (laion2b), 512-d  [pip install open_clip_torch]
      resnet  torchvision ResNet-50 (ImageNet), 2048-d
      none    no embedding (cheap signals only; for smoke tests)
  * cheap signals: motion (mean |diff| to the previous sampled frame), shot
    change (HSV histogram distance), brightness, contrast, sharpness, and
    saturation -- mean over the two frames, plus the max of motion / shot.

Frames are decoded sequentially (no seeking) and downscaled on arrival, so a
286 s video never holds more than one chunk of full frames in memory.
逐秒两帧：冻结图像编码器 + 运动 / 镜头切换 / 亮度等低成本特征。

    # train / val clips (rows of build_train_clips.py's output)
    python3 tools/extract_feats.py --gt data/train.with_video.jsonl --out-dir outputs/feats
    # test videos
    python3 tools/extract_feats.py --index baseline/test_index.json \
        --video-dir baseline/video --out-dir outputs/feats_test
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
from build_sft_temporal import step_bounds  # noqa: E402
from decode_scores import make_windows, read_jsonl  # noqa: E402
from baseline_qwen import video_meta  # noqa: E402
from reframe import default_device, iter_frames  # noqa: E402
from audit_videos import load_index as load_any_index  # noqa: E402  (JSON or JSONL)

LOW_NAMES = ["motion", "shot", "brightness", "contrast", "sharpness", "saturation",
             "motion_max", "shot_max"]
ENC_SHORT_SIDE = 256  # frames are shrunk to this before the encoder's own resize
CHUNK = 64


class Encoder:
    """Frozen image encoder: list of BGR frames -> (n, dim) float32."""

    def __init__(self, name, device):
        self.name, self.device, self.dim = name, device, 0
        if name == "none":
            return
        import torch
        self.torch = torch
        if name == "clip":
            try:
                import open_clip
            except ImportError:
                sys.exit("[fatal] --backbone clip needs: pip install open_clip_torch")
            model, _, self.pre = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            self.model, self.dim = model.eval().to(device), 512
        elif name == "resnet":
            import torchvision
            w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model = torchvision.models.resnet50(weights=w)
            model.fc = torch.nn.Identity()
            self.model, self.pre, self.dim = model.eval().to(device), w.transforms(), 2048
        else:
            sys.exit("[fatal] unknown backbone %s" % name)

    def __call__(self, frames):
        if self.dim == 0 or not frames:
            return np.zeros((len(frames), 0), dtype=np.float32)
        from PIL import Image
        torch = self.torch
        x = torch.stack([self.pre(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                         for f in frames]).to(self.device)
        with torch.no_grad():
            if self.name == "clip":
                e = self.model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
            else:
                e = self.model(x)
        return e.float().cpu().numpy()


def step_times(windows):
    """Per global step: (t at 1/4, t at 3/4) in seconds."""
    out = []
    for s, e, k in windows:
        for j in range(k):
            a, b = step_bounds(j, k, e - s)
            out.append((s + a + 0.25 * (b - a), s + a + 0.75 * (b - a)))
    return out


def shrink(frame, short=ENC_SHORT_SIDE):
    h, w = frame.shape[:2]
    scale = short / float(min(h, w))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def low_level(frame, prev):
    """Cheap per-frame signals; `prev` = (gray64, hist) of the previous sample."""
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    if prev is None:
        motion, shot = 0.0, 0.0
    else:
        motion = float(cv2.absdiff(gray, prev[0]).mean()) / 255.0
        shot = 1.0 - float(cv2.compareHist(hist, prev[1], cv2.HISTCMP_CORREL))
    lap = cv2.Laplacian(cv2.cvtColor(shrink(frame, 180), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    feats = [motion, shot, gray.mean() / 255.0, gray.std() / 128.0,
             float(np.log1p(lap)) / 10.0, hsv[..., 1].mean() / 255.0]
    return feats, (gray, hist)


def extract(path, n, fps, encoder):
    """-> (windows, emb (k, dim) float16, low (k, len(LOW_NAMES)) float32)."""
    windows = make_windows(n / float(fps))
    times = step_times(windows)
    idx = [[min(n - 1, max(0, int(round(t * fps)))) for t in pair] for pair in times]
    wanted = sorted({f for pair in idx for f in pair})
    emb, low, prev, chunk, order = {}, {}, None, [], []

    def flush():
        for f, e in zip(order, encoder(chunk)):
            emb[f] = e
        chunk.clear()
        order.clear()

    for f, fr in iter_frames(path, wanted):
        low[f], prev = low_level(fr, prev)
        chunk.append(shrink(fr))
        order.append(f)
        if len(chunk) == CHUNK:
            flush()
    if chunk:
        flush()
    if not low:
        raise RuntimeError("no frame decoded")
    # A frame that did not decode (e.g. past the real end of an overstated
    # count) borrows the nearest earlier decoded one.
    got = np.array(sorted(low))
    near = lambda f: int(got[max(0, np.searchsorted(got, f, side="right") - 1)])
    E = np.zeros((len(idx), encoder.dim), dtype=np.float32)
    Lw = np.zeros((len(idx), len(LOW_NAMES)), dtype=np.float32)
    for i, (f0, f1) in enumerate(idx):
        a, b = low[near(f0)], low[near(f1)]
        Lw[i, :6] = (np.asarray(a) + np.asarray(b)) / 2.0
        Lw[i, 6] = max(a[0], b[0])
        Lw[i, 7] = max(a[1], b[1])
        if encoder.dim:
            E[i] = (emb[near(f0)] + emb[near(f1)]) / 2.0
    return windows, E.astype(np.float16), Lw


def work_list(args):
    if args.gt:
        out = []
        for r in read_jsonl(args.gt):
            vp = r.get("video_path") or ""
            if os.path.exists(vp):
                out.append((r["video_id"], vp))
        return out
    return [(vid, os.path.join(args.video_dir, vid + ".mp4"))
            for vid, _ in load_any_index(args.index)]


def main():
    ap = argparse.ArgumentParser(description="Per-step features for the light temporal head")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt", help="train.with_video.jsonl (rows with video_path)")
    src.add_argument("--index", help="test_index.json (with --video-dir)")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbone", default="clip", choices=["clip", "resnet", "none"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.index and not args.video_dir:
        ap.error("--index needs --video-dir")

    work = work_list(args)
    if args.limit:
        work = work[:args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    encoder = Encoder(args.backbone, args.device or default_device())
    print("backbone %s (dim %d) on %s, %d videos" % (args.backbone, encoder.dim,
                                                     encoder.device, len(work)))
    t0, done, failed = time.time(), 0, 0
    for i, (vid, path) in enumerate(work):
        out = os.path.join(args.out_dir, vid + ".npz")
        if os.path.exists(out):
            continue
        n, fps, W, H = video_meta(path) if os.path.exists(path) else (0, 0.0, 0, 0)
        if n <= 0 or W <= 0:
            print("[warn] %s: unreadable, skipped" % vid)
            failed += 1
            continue
        try:
            windows, E, Lw = extract(path, n, fps, encoder)
        except Exception as e:  # one bad video must not stop the batch
            print("[warn] %s: %s" % (vid, e))
            failed += 1
            continue
        meta = {"video_id": vid, "fps": fps, "n_frames": n, "width": W, "height": H,
                "windows": [list(w) for w in windows], "backbone": args.backbone,
                "low_names": LOW_NAMES}
        np.savez_compressed(out, emb=E, low=Lw, meta=json.dumps(meta))
        done += 1
        if (i + 1) % 25 == 0:
            print("  %d/%d  %.0fs" % (i + 1, len(work), time.time() - t0))
    print("features -> %s  (new %d, failed %d, %.0fs)" % (args.out_dir, done, failed,
                                                         time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
