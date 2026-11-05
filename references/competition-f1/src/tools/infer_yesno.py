#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-shot per-second highlight scores from a frozen VLM, REZE-style / 逐秒是非打分（零样本）.

REZE (arXiv 2608.04480) gets per-clip saliency from an untouched VLM by asking a
yes/no question about a short clip and reading the Yes/No next-token logits,
instead of asking the model to write timestamps; aggregation is left to plain
code. Here every one-second step of the usual layout (decode_scores.make_windows
+ build_sft_temporal.step_bounds) gets a `--clip-sec` context centered on it,
sampled at `--clip-fps`, and

    p = sigmoid(logsumexp(Yes variants) - logsumexp(No variants))

at the first answer position. One forward pass per step, no generation, no
training. Records carry `p_sec` only (like train_temporal_head.py), so
decode_scores.py, ensemble_probs.py and the teacher / pseudo-label steps read
them unchanged.
冻结 VLM 对每秒（带前后上下文）回答是/否，取 Yes/No logit 作为高光概率；
输出与其他 probs 同格式。用途：评估零样本老师、给未标注视频打伪标签。

Any transformers image-text-to-text chat model should work (Qwen2.5-VL,
Qwen3-VL, Qwen3.5, InternVL3-hf ...): frames go in as a list of images resized
to `--max-side`, which also bounds the token count across model families.

    # all 987 labeled clips (no training -> every row is held-out)
    python tools/infer_yesno.py --model /models/Qwen3-VL-8B-Instruct \
        --gt data/train.with_video.jsonl --out outputs/yesno_qwen3vl8b.probs.jsonl
    python tools/decode_scores.py --probs outputs/yesno_qwen3vl8b.probs.jsonl --split all --tune
    # model-free checks (CPU)
    python tools/infer_yesno.py --selfcheck
"""

import os
import sys
import json
import math
import time
import argparse

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baseline"))
from build_sft_temporal import step_bounds  # noqa: E402
from decode_scores import MAX_WINDOW_SEC, make_windows, step_centers  # noqa: E402

DEFAULT_QUESTION = (
    "These {n} frames are sampled from a {sec:g}-second moment of a longer video. "
    "Is this moment a highlight that should be kept in a short highlight edit of the video? "
    "Answer Yes or No.")
DIGIT_QUESTION = (
    "These {n} frames are sampled from a {sec:g}-second moment of a longer video. "
    "On a scale of 0 to 9, how strongly should this moment be kept in a short highlight edit "
    "of the video (0 = boring filler, 9 = the best moment)? Answer with a single digit.")
YES_WORDS = ("Yes", "yes", " Yes", " yes", "YES")
NO_WORDS = ("No", "no", " No", " no", "NO")
DEFAULT_FEAT_LAYERS = "0.33,0.56,0.78,1.0"


# ------------------------------- pure helpers -------------------------------
def step_spans(windows):
    """Global [start, end) seconds of every step in the window layout."""
    out = []
    for s, e, k in windows:
        for j in range(k):
            a, b = step_bounds(j, k, e - s)
            out.append((s + a, s + b))
    return out


def clip_times(center, clip_sec, clip_fps, duration):
    """Sample times of a clip_sec context around `center`, shifted to stay
    inside [0, duration]; shorter videos use the whole video."""
    n = max(1, int(round(clip_sec * clip_fps)))
    span = min(clip_sec, duration)
    start = min(max(0.0, center - span / 2.0), max(0.0, duration - span))
    step = span / n
    return [start + (i + 0.5) * step for i in range(n)]


def time_to_frame(t, fps, n_frames):
    return min(n_frames - 1, max(0, int(math.floor(t * fps))))


def token_ids(tokenizer, words):
    """First token id of each spelling, deduplicated, order kept."""
    ids = []
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def yes_prob(logits, yes_ids, no_ids):
    """logits [batch, vocab] -> (p_yes, margin) with p = sigmoid(margin)."""
    lg = np.asarray(logits, dtype=np.float64)

    def lse(ids):
        x = lg[:, ids]
        m = x.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]

    margin = lse(yes_ids) - lse(no_ids)
    return 1.0 / (1.0 + np.exp(-margin)), margin


def digit_prob(logits, digit_ids):
    """logits [batch, vocab] -> (p, margin): softmax over the ten digit tokens,
    expected digit e in [0, 9], p = (e + 0.5) / 10 so it never hits 0 or 1."""
    lg = np.asarray(logits, dtype=np.float64)[:, digit_ids]
    lg = lg - lg.max(axis=1, keepdims=True)
    w = np.exp(lg)
    w /= w.sum(axis=1, keepdims=True)
    p = (w @ np.arange(10.0) + 0.5) / 10.0
    return p, np.log(p / (1.0 - p))


def feat_layer_ids(fracs, n_layers):
    """Fractions of depth -> sorted distinct 0-based decoder layer indices."""
    return sorted({min(n_layers - 1, max(0, int(round(f * n_layers)) - 1)) for f in fracs})


def read_frames(path, frame_ids, max_side, portrait_crop=False):
    """{frame id: RGB uint8 array, longest side <= max_side}; sequential decode
    (no seeking, see reframe.iter_frames). portrait_crop: training augmentation for the portrait test videos --
    keep the centered 9:16 part of a landscape frame and scale it to max_side (upscaling the low-res QVH clips)."""
    want = set(frame_ids)
    out = {}
    if not want:
        return out
    last = max(want)
    cap = cv2.VideoCapture(path)
    try:
        idx = 0
        while idx <= last:
            if not cap.grab():
                break
            if idx in want:
                ok, fr = cap.retrieve()
                if ok:
                    h, w = fr.shape[:2]
                    if portrait_crop and w > h:
                        cw = int(h * 9 / 16)
                        fr = fr[:, (w - cw) // 2:(w - cw) // 2 + cw]
                        h, w = fr.shape[:2]
                        s = max_side / float(max(h, w))
                        fr = cv2.resize(fr, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_CUBIC)
                        out[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        idx += 1
                        continue
                    s = max_side / float(max(h, w))
                    if s < 1.0:
                        fr = cv2.resize(fr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                                        interpolation=cv2.INTER_AREA)
                    out[idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
    return out


def video_info(path):
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return n, float(fps), W, H


def score_video(path, scorer, args, prepared=None):
    """-> record fields (fps, n_frames, size, windows, p_sec, yes_margin).
    `prepared` is prepare_video()'s result when the frames were decoded ahead."""
    n, fps, W, H, windows, items = prepared if prepared is not None else prepare_video(path, args)
    p, m, feats = [], [], {}
    for i in range(0, len(items), args.batch):
        res = scorer(items[i:i + args.batch], args.clip_sec)
        p.extend(res[0])
        m.extend(res[1])
        # optional third value: {name: [batch, layers, dim]} hidden-state features
        for k, v in (res[2] if len(res) > 2 and res[2] else {}).items():
            feats.setdefault(k, []).append(v)
    rec = {"fps": fps, "n_frames": n, "width": W, "height": H,
           "windows": [[round(s, 4), round(e, 4), k] for s, e, k in windows],
           "p_sec": [round(float(v), 4) for v in p],
           "yes_margin": [round(float(v), 3) for v in m]}
    if feats:
        rec["_feats"] = {k: np.concatenate(v, axis=0) for k, v in feats.items()}
    return rec


def prepare_video(path, args):
    """Decode everything one video needs -> (n, fps, W, H, windows, per-step frame lists).
    CPU only, so the main loop runs it for the next video while the GPU scores this one."""
    n, fps, W, H = video_info(path)
    if n <= 1 or W <= 0:
        raise IOError("cannot read video: %s" % path)
    duration = n / fps
    windows = make_windows(duration, args.max_window_sec)
    spans = step_spans(windows)
    centers = step_centers(windows)
    times = [clip_times(c, args.clip_sec, args.clip_fps, duration) for c in centers]
    ids = [[time_to_frame(t, fps, n) for t in ts] for ts in times]
    frames = read_frames(path, [f for row in ids for f in row], args.max_side, args.portrait_crop)
    if not frames:
        raise IOError("no frames decoded: %s" % path)
    have = sorted(frames)
    items = []
    for (a, b), row in zip(spans, ids):
        # a frame past the real end (bad FRAME_COUNT) -> nearest decoded one
        imgs = [frames[f] if f in frames else frames[have[int(np.argmin(np.abs(np.array(have) - f)))]]
                for f in row]
        items.append(imgs)
    return n, fps, W, H, windows, items


# ---------------------------------- model -----------------------------------
class VLMScorer:
    """Frozen VLM: list of frame lists -> (p, margin[, feats]) per item.

    mode "yesno": REZE Yes/No margin; mode "digits": expected 0-9 rating.
    feat_layers: depth fractions; when set, forward hooks on those decoder
    layers keep, per item, the answer-position hidden state ("last") and the
    mean over image tokens ("vis") -- the dense head's input, from the same
    forward pass as the score. 同一次前向顺便取隐藏状态特征。
    """

    def __init__(self, model_path, device, quant4, question, mode="yesno", feat_layers=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.question, self.mode = torch, question, mode
        kw = {}
        if quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        dtype = (torch.bfloat16 if device.startswith("cuda")
                 else torch.float16 if device == "mps" else torch.float32)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=dtype, device_map={"": device}, attn_implementation="sdpa", **kw)
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.proc.tokenizer.padding_side = "left"
        # do_resize explicit: some checkpoints (TimeLens-8B) ship do_resize in
        # tokenizer_config.json, which leaks into the image kwargs and skips the
        # resize the patch grid assumes. InternVL: one 448 px tile per frame
        # (256 tokens) instead of dynamic tiling (768+ per frame).
        # 显式 do_resize；InternVL 关闭动态切块。
        self.images_kwargs = {"do_resize": True}
        if hasattr(getattr(self.proc, "image_processor", None), "crop_to_patches"):
            self.images_kwargs["crop_to_patches"] = False
        self.device = device
        tok = self.proc.tokenizer
        self.yes_ids, self.no_ids = token_ids(tok, YES_WORDS), token_ids(tok, NO_WORDS)
        if not self.yes_ids or not self.no_ids or set(self.yes_ids) & set(self.no_ids):
            raise ValueError("bad Yes/No token ids: %s / %s" % (self.yes_ids, self.no_ids))
        self.digit_ids = [token_ids(tok, [str(d)])[0] for d in range(10)]
        if len(set(self.digit_ids)) != 10:
            raise ValueError("digit tokens are not distinct: %s" % self.digit_ids)
        self.layer_ids, self._cap, self._mask = [], {}, None
        if feat_layers:
            self._hook_layers(feat_layers)

    def _hook_layers(self, fracs):
        torch = self.torch
        best = None
        for name, mod in self.model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and name.endswith("layers")
                    and "vision" not in name and "visual" not in name):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        if best is None:
            raise ValueError("no decoder layer list found for feature hooks")
        cfg = self.model.config
        tid = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        if tid is None:
            tid = self.proc.tokenizer.convert_tokens_to_ids(getattr(self.proc, "image_token", "<|image_pad|>"))
        self.image_token_id = int(tid)
        self.layer_ids = feat_layer_ids(fracs, len(best[1]))
        self.layer_path = "%s (%d layers)" % (best[0], len(best[1]))

        def make_hook(i):
            def hook(_module, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                mask = self._mask.to(h.dtype).unsqueeze(-1)
                vis = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                self._cap[i] = (h[:, -1, :].float().cpu().numpy(), vis.float().cpu().numpy())
            return hook
        for i in self.layer_ids:
            best[1][i].register_forward_hook(make_hook(i))

    def prompt(self, n_frames, clip_sec):
        content = [{"type": "image"} for _ in range(n_frames)]
        content.append({"type": "text", "text": self.question.format(n=n_frames, sec=clip_sec)})
        # enable_thinking=False: Qwen3.x would otherwise open a <think> block;
        # templates without the flag ignore it. 关闭思考，下一个 token 即答案。
        return self.proc.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)

    def __call__(self, items, clip_sec):
        torch = self.torch
        texts = [self.prompt(len(imgs), clip_sec) for imgs in items]
        # padding must go in text_kwargs: transformers 5 silently drops a bare
        # padding=True, and a batch of unequal prompts then fails to tensorize.
        # 新版 transformers 会忽略裸 padding 参数，需放进 text_kwargs。
        inputs = self.proc(text=texts, images=[list(imgs) for imgs in items],
                           text_kwargs={"padding": True}, images_kwargs=dict(self.images_kwargs),
                           return_tensors="pt").to(self.device)
        self._cap = {}
        if self.layer_ids:
            self._mask = inputs["input_ids"] == self.image_token_id
        with torch.no_grad():
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
        last = out.logits[:, -1, :].float().cpu().numpy()
        if self.mode == "digits":
            p, m = digit_prob(last, self.digit_ids)
        else:
            p, m = yes_prob(last, self.yes_ids, self.no_ids)
        if not self.layer_ids:
            return p.tolist(), m.tolist()
        feats = {"last": np.stack([self._cap[i][0] for i in self.layer_ids], axis=1).astype(np.float16),
                 "vis": np.stack([self._cap[i][1] for i in self.layer_ids], axis=1).astype(np.float16)}
        return p.tolist(), m.tolist(), feats


# --------------------------------- inputs -----------------------------------
def load_jobs(args):
    """-> [(video_id, [tw, th], path)] from --gt rows or --index + --video-dir."""
    jobs = []
    if args.gt:
        with open(args.gt, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    jobs.append((r["video_id"], r["targetRatioWH"], r.get("video_path") or ""))
    else:
        from audit_videos import load_index as load_any_index
        for vid, ratio in load_any_index(args.index):  # -> (video_id, [tw, th]) tuples
            jobs.append((vid, ratio, os.path.join(args.video_dir, vid + ".mp4")))
    return jobs[:args.num_videos] if args.num_videos > 0 else jobs


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        ids.add(r["video_id"])
    return ids


# -------------------------------- selfcheck ---------------------------------
def selfcheck(tmp_dir):
    """Model-free checks: layout, sampling, Yes/No math, end-to-end with a fake
    scorer on a synthetic video, and that decode_scores reads the record."""
    import decode_scores as ds
    fails = []

    def check(name, ok, extra=""):
        print("  [%s] %s %s" % ("PASS" if ok else "FAIL", name, extra))
        if not ok:
            fails.append(name)

    for dur in (0.7, 5.0, 12.4, 31.9, 32.0, 75.3, 286.0):
        win = make_windows(dur)
        sp, ce = step_spans(win), step_centers(win)
        ok = (len(sp) == sum(k for _, _, k in win) == len(ce)
              and all(a <= c < b + 1e-9 for (a, b), c in zip(sp, ce))
              and abs(sp[-1][1] - dur) < 1e-6 and sp[0][0] == 0.0)
        check("layout %.1fs: %d steps, centers inside spans" % (dur, len(sp)), ok)

    ts = clip_times(10.5, 3.0, 2.0, 20.0)
    check("clip 3s@2fps centered", len(ts) == 6 and abs(np.mean(ts) - 10.5) < 1e-9, str(ts))
    ts = clip_times(0.5, 3.0, 2.0, 20.0)
    check("clip shifted inside at the start", ts[0] >= 0 and abs(ts[-1] - ts[0] - 2.5) < 1e-9)
    ts = clip_times(19.5, 3.0, 2.0, 20.0)
    check("clip shifted inside at the end", ts[-1] <= 20.0 and ts[0] >= 17.0 - 1e-9)
    ts = clip_times(0.3, 3.0, 2.0, 0.8)
    check("clip on a 0.8 s video stays inside", all(0 <= t <= 0.8 for t in ts))

    lg = np.full((3, 20), -5.0)
    lg[0, 3], lg[0, 7] = 4.0, 0.0      # yes >> no
    lg[1, 3], lg[1, 7] = 0.0, 4.0      # no >> yes
    lg[2, 3], lg[2, 4], lg[2, 7] = 1.0, 1.0, 1.0 + math.log(2.0)  # yes variants sum = no
    p, m = yes_prob(lg, [3, 4], [7])
    check("yes_prob: yes / no / tie across variants",
          p[0] > 0.98 and p[1] < 0.02 and abs(p[2] - 0.5) < 1e-9, str(np.round(p, 4)))

    dl = np.full((2, 20), -9.0)
    dl[0, 19] = 9.0                     # all mass on digit 9
    dl[1, 10:20] = 0.0                  # uniform over 0..9 -> e = 4.5
    p, m = digit_prob(dl, list(range(10, 20)))
    check("digit_prob: peak at 9 / uniform", abs(p[0] - 0.95) < 1e-3 and abs(p[1] - 0.5) < 1e-9
          and np.all(np.isfinite(m)), str(np.round(p, 4)))
    check("feat_layer_ids 36 layers", feat_layer_ids([0.33, 0.56, 0.78, 1.0], 36) == [11, 19, 27, 35])
    check("feat_layer_ids 28 layers dedup", feat_layer_ids([0.33, 0.335, 1.0], 28) == [8, 27])

    class Tok:
        def encode(self, w, add_special_tokens=False):
            return {"Yes": [1], "yes": [2], " Yes": [3, 9], " yes": [2], "YES": [1, 1],
                    "No": [5], "no": [6], " No": [5], " no": [7], "NO": []}[w]
    check("token_ids dedup + first token", token_ids(Tok(), YES_WORDS) == [1, 2, 3]
          and token_ids(Tok(), NO_WORDS) == [5, 6, 7])

    # synthetic 7.3 s video: brightness ramps up, so a brightness scorer ranks late steps higher
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, "yesno_selfcheck.mp4")
    fps, n = 25.0, 183
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for i in range(n):
        vw.write(np.full((180, 320, 3), int(20 + 200 * i / (n - 1)), np.uint8))
    vw.release()

    class Fake:
        def __call__(self, items, clip_sec):
            b = [float(np.mean([im.mean() for im in imgs])) / 255.0 for imgs in items]
            f = np.tile(np.array(b, np.float16)[:, None, None], (1, 2, 3))
            return b, [math.log(v / (1 - v)) for v in b], {"last": f, "vis": f}

    class A:
        max_window_sec, clip_sec, clip_fps, max_side, batch, portrait_crop = MAX_WINDOW_SEC, 3.0, 2.0, 64, 4, False
    rec = score_video(path, Fake(), A())
    rec.update(video_id="x", targetRatioWH=[9, 16])
    k = sum(w[2] for w in rec["windows"])
    check("record: one p per step", len(rec["p_sec"]) == k == len(rec["yes_margin"]), "(%d)" % k)
    feats = rec.pop("_feats", {})
    check("record: features one row per step across batches",
          set(feats) == {"last", "vis"} and feats["last"].shape == (k, 2, 3)
          and np.allclose(feats["vis"][:, 0, 0], rec["p_sec"], atol=2e-3))
    check("record: fake brightness scores increase", all(np.diff(rec["p_sec"]) >= -1e-3))
    steps = ds.frame_steps([tuple(w) for w in rec["windows"]], rec["fps"], rec["n_frames"])
    pp = ds.step_probs(rec, 2.0)
    check("decode_scores reads it (p_sec path, temp)",
          len(pp) == k and steps.max() == k - 1 and np.all((pp > 0) & (pp < 1)))
    os.remove(path)
    print("  %d failed" % len(fails))
    return 1 if fails else 0


# ---------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Zero-shot yes/no per-second highlight scores")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--gt", help="train.with_video.jsonl (rows with video_path)")
    src.add_argument("--index", help="test_index.json (with --video-dir)")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--model", help="local VLM dir (transformers image-text-to-text)")
    ap.add_argument("--out", help="*.probs.jsonl")
    ap.add_argument("--mode", choices=("yesno", "digits"), default="yesno",
                    help="yesno: REZE Yes/No margin; digits: expected 0-9 rating")
    ap.add_argument("--question", default=None,
                    help="prompt; {n} = frames, {sec} = clip seconds (default depends on --mode)")
    ap.add_argument("--save-feats", default=None,
                    help="dir for per-video <video_id>.npz hidden-state features (dense head input)")
    ap.add_argument("--feat-layers", default=DEFAULT_FEAT_LAYERS,
                    help="decoder depth fractions hooked for --save-feats")
    ap.add_argument("--portrait-crop", action="store_true", help="augmentation: centered 9:16 crop of landscape frames")
    ap.add_argument("--clip-sec", type=float, default=3.0, help="context per step (REZE: 3 s)")
    ap.add_argument("--clip-fps", type=float, default=2.0, help="frames per second of context")
    ap.add_argument("--max-side", type=int, default=448, help="longest image side fed to the VLM")
    ap.add_argument("--max-window-sec", type=float, default=MAX_WINDOW_SEC,
                    help="output layout only (matches infer_scores.py)")
    ap.add_argument("--batch", type=int, default=8, help="steps per forward pass")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quant4", action="store_true", help="4-bit NF4 weights (bitsandbytes)")
    ap.add_argument("--num-videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="append, skipping ids already scored")
    ap.add_argument("--selfcheck", action="store_true", help="model-free checks, then exit")
    ap.add_argument("--tmp-dir", default="outputs/tmp")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck(args.tmp_dir)
    if not (args.model and args.out and (args.gt or (args.index and args.video_dir))):
        ap.error("need --model, --out and --gt or --index + --video-dir")

    if args.question is None:
        args.question = DIGIT_QUESTION if args.mode == "digits" else DEFAULT_QUESTION
    jobs = load_jobs(args)
    skip = done_ids(args.out) if args.resume else set()
    todo = [j for j in jobs if j[0] not in skip]
    print("videos: %d  (already scored %d)  model %s" % (len(todo), len(jobs) - len(todo), args.model))
    fracs = [float(x) for x in args.feat_layers.split(",")] if args.save_feats else None
    scorer = VLMScorer(args.model, args.device, args.quant4, args.question, args.mode, fracs)
    print("yes ids %s  no ids %s  digit ids %s" % (scorer.yes_ids, scorer.no_ids, scorer.digit_ids))
    if args.save_feats:
        os.makedirs(args.save_feats, exist_ok=True)
        print("feature hooks: %s -> layers %s" % (scorer.layer_path, scorer.layer_ids))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"mode": args.mode, "model": os.path.basename(os.path.normpath(args.model)),
            "question": args.question, "clip_sec": args.clip_sec, "clip_fps": args.clip_fps,
            "max_side": args.max_side, "quant4": args.quant4,
            "feat_layers": scorer.layer_ids if args.save_feats else None}
    stats, t0 = {"videos": 0, "steps": 0, "failed": 0}, time.time()
    # decode the next video on a worker thread while the GPU scores the current one
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(prepare_video, todo[0][2], args) if todo else None
    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as fout:
        for i, (vid, ratio, path) in enumerate(todo, 1):
            rec = {"video_id": vid, "targetRatioWH": [int(v) for v in ratio], "scorer": meta}
            cur = nxt
            nxt = pool.submit(prepare_video, todo[i][2], args) if i < len(todo) else None
            try:
                rec.update(score_video(path, scorer, args, cur.result()))
                feats = rec.pop("_feats", None)
                if args.save_feats and feats:
                    np.savez(os.path.join(args.save_feats, vid + ".npz"),
                             p_sec=np.asarray(rec["p_sec"], np.float32),
                             margin=np.asarray(rec["yes_margin"], np.float32), **feats)
                stats["steps"] += len(rec["p_sec"])
                msg = "%d steps  p %.2f..%.2f" % (len(rec["p_sec"]), min(rec["p_sec"]), max(rec["p_sec"]))
            except Exception as e:  # noqa: BLE001  one bad video must not stop the batch
                stats["failed"] += 1
                rec["error"] = "%s: %s" % (type(e).__name__, e)
                msg = "[error] %s" % rec["error"]
            stats["videos"] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            el = time.time() - t0
            print("  [%d/%d] %s %s  (%.2f s/step)"
                  % (i, len(todo), vid, msg, el / max(stats["steps"], 1)), flush=True)
    print("done: %s in %.1f min -> %s" % (stats, (time.time() - t0) / 60, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
