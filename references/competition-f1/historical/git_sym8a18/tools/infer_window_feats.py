#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whole-window features from a frozen VLM / 整段一起看的逐秒特征.

infer_yesno.py shows the VLM one 3 s context per step, so a step's features
never see the rest of the clip. The hosted teachers (api_teacher.py) look at
the whole clip at once and rank seconds against each other; heads trained to
imitate them on the step-wise features stayed far below the teachers.

Here one forward pass covers a whole window (<= 32 s): one frame per step at
the step center, each preceded by a "第k秒画面" tag, then an instruction and
one readout query per step ("第k秒：", the colon token is the readout). The
decoder is causal, so the queries -- placed after every frame -- attend to the
whole window, like a teacher writing its per-second scores. Per step and per
hooked layer this keeps
  last  hidden state at the step's readout token (has seen the whole window)
  vis   mean hidden state over the step frame's image tokens (sees the past)
in the npz layout infer_yesno.py --save-feats writes (p_sec / margin are zero
placeholders), so train_vlm_head.py reads it unchanged.
一次前向看完整个窗口；每秒的读出 token 放在所有画面之后，能看到全段。

    python tools/infer_window_feats.py --model $TIMELENS --gt data/train.with_video.jsonl \\
        --out outputs/wf_timelens8b.probs.jsonl --save-feats outputs/wff_timelens8b --resume
    python tools/infer_window_feats.py --model $TIMELENS --index $T/submission_index.jsonl --video-dir $T/video \\
        --out outputs/wf_test_timelens8b.probs.jsonl --save-feats outputs/wff_test_timelens8b --resume
"""

import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decode_scores import MAX_WINDOW_SEC, make_windows, step_centers  # noqa: E402
from infer_yesno import (DEFAULT_FEAT_LAYERS, done_ids, feat_layer_ids, load_jobs, read_frames,  # noqa: E402
                         time_to_frame, video_info)

INSTRUCTION = "以上是一段视频按每秒 1 帧抽取的画面。下面逐秒判断每一秒是否值得剪进这段视频的高光短片。\n"
READ_MARK = "："


def prepare(path, args):
    """-> (n, fps, W, H, windows, [[RGB frame per step] per window])"""
    n, fps, W, H = video_info(path)
    if n <= 1 or W <= 0:
        raise IOError("cannot read video: %s" % path)
    windows = make_windows(n / fps, args.max_window_sec)
    centers = step_centers(windows)
    ids = [time_to_frame(t, fps, n) for t in centers]
    frames = read_frames(path, ids, args.max_side)
    if not frames:
        raise IOError("no frames decoded: %s" % path)
    have = sorted(frames)
    per_step = [frames[f] if f in frames else frames[have[int(np.argmin(np.abs(np.array(have) - f)))]] for f in ids]
    out, off = [], 0
    for _, _, k in windows:
        out.append(per_step[off:off + k])
        off += k
    return n, fps, W, H, windows, out


class WindowEncoder:
    def __init__(self, model_path, device, fracs):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.device = torch, device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            device_map={"": device}, attn_implementation="sdpa")
        self.model.eval()
        self.proc = AutoProcessor.from_pretrained(model_path)
        self.images_kwargs = {"do_resize": True}
        if hasattr(getattr(self.proc, "image_processor", None), "crop_to_patches"):
            self.images_kwargs["crop_to_patches"] = False
        cfg = self.model.config
        tid = getattr(cfg, "image_token_id", None) or getattr(cfg, "image_token_index", None)
        self.image_token_id = int(tid)
        best = None
        for name, mod in self.model.named_modules():
            if (isinstance(mod, torch.nn.ModuleList) and name.endswith("layers")
                    and "vision" not in name and "visual" not in name):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
        self.layer_ids = feat_layer_ids(fracs, len(best[1]))
        self.layer_path = "%s (%d layers)" % (best[0], len(best[1]))
        self._spans, self._reads, self._cap = None, None, {}

        def make_hook(i):
            def hook(_m, _inp, out):
                h = (out[0] if isinstance(out, (tuple, list)) else out)[0]          # [T, D]
                vis = torch.stack([h[a:b].float().mean(0) for a, b in self._spans])  # [k, D]
                last = h[self._reads].float()                                        # [k, D]
                self._cap[i] = (last.cpu().numpy(), vis.cpu().numpy())
            return hook
        for i in self.layer_ids:
            best[1][i].register_forward_hook(make_hook(i))

    def prompt(self, k):
        content = []
        for j in range(k):
            content.append({"type": "text", "text": "第%d秒画面" % j})
            content.append({"type": "image"})
        content.append({"type": "text", "text": INSTRUCTION + "".join("第%d秒%s\n" % (j, READ_MARK) for j in range(k))})
        return self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=False, enable_thinking=False)

    def __call__(self, imgs):
        """frames of one window -> (last [k, L, D], vis [k, L, D])"""
        torch, k = self.torch, len(imgs)
        inputs = self.proc(text=[self.prompt(k)], images=[list(imgs)], images_kwargs=self.images_kwargs,
                           return_tensors="pt").to(self.device)
        ids = inputs["input_ids"][0].tolist()
        spans, i = [], 0
        while i < len(ids):
            if ids[i] == self.image_token_id:
                j = i
                while j < len(ids) and ids[j] == self.image_token_id:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        if len(spans) != k:
            raise ValueError("found %d image spans for %d frames" % (len(spans), k))
        tok = self.proc.tokenizer
        reads = [p for p in range(spans[-1][1], len(ids)) if READ_MARK in tok.decode([ids[p]])]
        if len(reads) != k:
            raise ValueError("found %d readout tokens for %d steps" % (len(reads), k))
        self._spans, self._reads, self._cap = spans, reads, {}
        with torch.no_grad():
            try:
                self.model(**inputs, logits_to_keep=1)
            except TypeError:
                self.model(**inputs)
        last = np.stack([self._cap[l][0] for l in self.layer_ids], axis=1).astype(np.float16)
        vis = np.stack([self._cap[l][1] for l in self.layer_ids], axis=1).astype(np.float16)
        return last, vis


def main():
    ap = argparse.ArgumentParser(description="Whole-window per-step features from a frozen VLM")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gt", help="train.with_video.jsonl (rows with video_path)")
    src.add_argument("--index", help="test index (with --video-dir)")
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="*.probs.jsonl (layout records; p_sec = 0)")
    ap.add_argument("--save-feats", required=True)
    ap.add_argument("--feat-layers", default=DEFAULT_FEAT_LAYERS)
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--max-window-sec", type=float, default=MAX_WINDOW_SEC)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    jobs = load_jobs(args)
    skip = done_ids(args.out) if args.resume else set()
    todo = [j for j in jobs if j[0] not in skip]
    print("videos: %d (already done %d)" % (len(todo), len(jobs) - len(todo)), flush=True)
    enc = WindowEncoder(args.model, args.device, [float(x) for x in args.feat_layers.split(",")])
    print("hooks: %s -> layers %s" % (enc.layer_path, enc.layer_ids), flush=True)
    os.makedirs(args.save_feats, exist_ok=True)
    meta = {"mode": "window", "model": os.path.basename(os.path.normpath(args.model)), "max_side": args.max_side,
            "feat_layers": enc.layer_ids, "instruction": INSTRUCTION}
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(prepare, todo[0][2], args) if todo else None
    t0, steps = time.time(), 0
    with open(args.out, "a" if args.resume else "w", encoding="utf-8") as fout:
        for i, (vid, ratio, path) in enumerate(todo, 1):
            cur = nxt
            nxt = pool.submit(prepare, todo[i][2], args) if i < len(todo) else None
            rec = {"video_id": vid, "targetRatioWH": [int(v) for v in ratio], "scorer": meta}
            try:
                n, fps, W, H, windows, per_win = cur.result()
                L, V = [], []
                for imgs in per_win:
                    last, vis = enc(imgs)
                    L.append(last)
                    V.append(vis)
                last, vis = np.concatenate(L), np.concatenate(V)
                k = len(last)
                rec.update({"fps": fps, "n_frames": n, "width": W, "height": H,
                            "windows": [[round(s, 4), round(e, 4), kk] for s, e, kk in windows],
                            "p_sec": [0.0] * k, "yes_margin": [0.0] * k})
                np.savez(os.path.join(args.save_feats, vid + ".npz"), p_sec=np.zeros(k, np.float32),
                         margin=np.zeros(k, np.float32), last=last, vis=vis)
                steps += k
                msg = "%d steps, %d windows" % (k, len(windows))
            except Exception as e:  # noqa: BLE001  one bad video must not stop the batch
                rec["error"] = "%s: %s" % (type(e).__name__, e)
                msg = "[error] " + rec["error"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print("  [%d/%d] %s %s (%.2f s/step)" % (i, len(todo), vid, msg, (time.time() - t0) / max(steps, 1)),
                  flush=True)
    print("done in %.1f min -> %s" % ((time.time() - t0) / 60, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
