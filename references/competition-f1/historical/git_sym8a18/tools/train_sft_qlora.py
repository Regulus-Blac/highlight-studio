#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA temporal SFT for Qwen2.5-VL on one 8 GB card / 单卡 8 GB 时序 SFT.

Trains on the chat rows from `build_sft_temporal.py`. Budget and constraints
are the ones in docs/ENV_SETUP.md: 4-bit NF4 LLM, frozen bf16 vision tower,
LoRA on the language model only, batch 1 + gradient accumulation, gradient
checkpointing, 16 frames at ~128 tokens each.

    python tools/train_sft_qlora.py \
        --model E:/AICTrain/models/Qwen2.5-VL-3B-Instruct \
        --train data/sft/sft_temporal_hl90_train.jsonl \
        --val   data/sft/sft_temporal_hl90_val.jsonl \
        --out   outputs/sft_qlora_hl90

Smoke test first (4 steps, no save): add `--limit 8 --grad-accum 2 --epochs 1`.
先跑冒烟测试。

Per-second scoring on Qwen3.5-4B (rows from `build_sft_temporal.py --task
scores`; the task is read from the rows, pixels default to the scoring budget):

    python tools/train_sft_qlora.py         --model E:/AICTrain/models/Qwen3.5-4B         --train data/sft/sft_scores_train.jsonl         --val   data/sft/sft_scores_val.jsonl         --out   outputs/sft_scores_q35 --epochs 2

bf16 LoRA on a 24 GB+ card (e.g. a rented RTX 5090), more pixels per frame,
and training-time augmentation -- flips, plus 9:16 "portrait" crops centered
on each clip's labeled subject, because the test set is mostly portrait
sources the 534x300 training clips never show:
bf16 LoRA + 翻转 / 竖屏模拟增强（5090 等 24 GB+ 显卡）：

    python tools/train_sft_qlora.py --model /models/Qwen3.5-4B --no-quant \
        --lora-r 64 --lora-alpha 128 --max-pixels 131072 --epochs 4 \
        --aug-flip 0.5 --aug-portrait 0.4 --spatial-src data/train.with_video.jsonl \
        --train data/sft/sft_scores_train.jsonl --val data/sft/sft_scores_val.jsonl \
        --out outputs/sft_scores_q35_v2 --num-workers 4

Serve the result with the same --no-quant / --max-pixels in infer_scores.py.

Only the answer tokens are supervised. The model is asked for logits of the
last few positions only (`logits_to_keep`), so the 151k-vocab logit tensor is
never materialized for the ~1100 prompt tokens -- that alone saves ~0.6 GB.
只监督答案 token，且只取末尾几位的 logits，省约 0.6 GB。

Outputs under --out: `epoch<k>/` LoRA adapters, `train_log.jsonl` (one line
per optimizer step plus one per validation pass), `args.json`.
"""

import os
import sys
import json
import math
import time
import random
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_vl as tv  # noqa: E402


def read_jsonl(path, limit=0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit > 0 else rows


def subject_centers(path):
    """video_id -> median crop center x (pixels) of its cropRois, if any."""
    out = {}
    if not path:
        return out
    for r in read_jsonl(path):
        rois = r.get("cropRois") or []
        if rois:
            out[r["video_id"]] = float(sorted(b[0] + b[2] / 2.0 for _, b in rois)[len(rois) // 2])
    return out


class SFTRows(torch.utils.data.Dataset):
    """Decode + tokenize lazily; a broken clip yields None and is skipped.

    `aug` = {"flip": p, "portrait": p, "centers": {video_id: cx_px}} draws a
    fresh augmentation per access (training only; None for val).
    """

    def __init__(self, rows, processor, nframes, aug=None):
        self.rows, self.processor, self.nframes, self.aug = rows, processor, nframes, aug

    def __len__(self):
        return len(self.rows)

    def draw_augment(self, vid):
        if not self.aug:
            return None
        out = {}
        if random.random() < self.aug["portrait"]:
            out["portrait"] = {"cx_px": self.aug["centers"].get(vid)}
        if random.random() < self.aug["flip"]:
            out["flip"] = True
        return out or None

    def __getitem__(self, i):
        r = self.rows[i]
        target = r["messages"][-1]["content"][0]["text"]
        try:
            inp = tv.build_inputs(self.processor, r["video"], r["duration"],
                                  target=target, nframes=self.nframes,
                                  task=r.get("task", "segments"),
                                  augment=self.draw_augment(r["video_id"]))
        except Exception as e:  # noqa: BLE001  report and skip the sample
            return {"error": "%s: %s" % (r.get("video_id"), e)}
        inp = dict(inp)
        inp["video_id"] = r["video_id"]
        return inp


def first(batch):
    return batch[0]


def sample_loss(model, inp):
    """Mean CE over the answer tokens of one sample. 单样本答案 token 的交叉熵。"""
    n_target, n_tail = inp["n_target"], inp["n_tail"]
    feed = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items()
            if k not in ("n_target", "n_tail", "video_id")}
    ids = feed["input_ids"][0]
    L = ids.shape[0]
    keep = n_target + n_tail + 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**feed, logits_to_keep=keep, use_cache=False)
    logits = out.logits[0, :n_target].float()           # predicts ids[L-keep+1 : ...]
    labels = ids[L - n_tail - n_target:L - n_tail]
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for inp in loader:
        if "error" in inp:
            continue
        tot += sample_loss(model, inp).item()
        n += 1
    model.train()
    return tot / max(n, 1), n


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main():
    ap = argparse.ArgumentParser(description="QLoRA temporal SFT (8 GB)")
    ap.add_argument("--model", required=True, help="Qwen2.5-VL-3B-Instruct dir")
    ap.add_argument("--train", default="data/sft/sft_temporal_hl90_train.jsonl")
    ap.add_argument("--val", default="data/sft/sft_temporal_hl90_val.jsonl")
    ap.add_argument("--out", default="outputs/sft_qlora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--nframes", type=int, default=tv.DEFAULT_NFRAMES)
    ap.add_argument("--max-pixels", type=int, default=None,
                    help="per-frame pixel budget (default depends on the task)")
    ap.add_argument("--quant-vision", action="store_true",
                    help="4-bit the vision tower too (saves ~1 GB, must match at inference)")
    ap.add_argument("--no-quant", action="store_true",
                    help="bf16 LoRA instead of 4-bit QLoRA (24 GB+ cards)")
    ap.add_argument("--aug-flip", type=float, default=0.0,
                    help="probability of a left-right flip per training sample")
    ap.add_argument("--aug-portrait", type=float, default=0.0,
                    help="probability of a 9:16 crop around the labeled subject")
    ap.add_argument("--spatial-src", default=None,
                    help="train.with_video.jsonl: cropRois give the portrait crop center")
    ap.add_argument("--resume-adapter", default=None,
                    help="continue from a saved adapter dir")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N train/val rows (smoke test)")
    ap.add_argument("--longest-first", action="store_true",
                    help="sort train rows longest first (smoke-test the memory peak)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = read_jsonl(args.train)
    if args.longest_first:
        train_rows.sort(key=lambda r: -r["duration"])
    train_rows = train_rows[:args.limit] if args.limit > 0 else train_rows
    val_rows = read_jsonl(args.val, args.limit) if args.val else []
    tasks = {r.get("task", "segments") for r in train_rows + val_rows}
    if len(tasks) != 1:
        raise SystemExit("mixed tasks in the data: %s" % sorted(tasks))
    task = tasks.pop()
    if args.max_pixels is None:
        args.max_pixels = tv.SCORE_MAX_PIXELS if task == "scores" else tv.DEFAULT_MAX_PIXELS
    min_pixels = tv.SCORE_MIN_PIXELS if task == "scores" else tv.DEFAULT_MIN_PIXELS
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    log_f = open(os.path.join(args.out, "train_log.jsonl"), "a", encoding="utf-8")

    def log(rec):
        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log_f.flush()

    processor = tv.load_processor(args.model, max_pixels=args.max_pixels,
                                  min_pixels=min_pixels)
    model = tv.load_model(args.model, adapter=args.resume_adapter,
                          quant_vision=args.quant_vision, for_training=True,
                          quant=not args.no_quant)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    if not args.resume_adapter:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                         lora_dropout=args.lora_dropout, bias="none",
                         target_modules=tv.LORA_TARGET_REGEX, task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    n_vis = sum(1 for n, p in model.named_parameters() if p.requires_grad and "visual" in n)
    assert n_vis == 0, "LoRA leaked into the vision tower"
    hit = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1]
                  for n, p in model.named_parameters() if p.requires_grad and ".lora_A" in n})
    print("LoRA on: %s" % ", ".join(hit))
    model.train()
    print("task %s | train rows %d | val rows %d | max_pixels %d"
          % (task, len(train_rows), len(val_rows), args.max_pixels))

    dl_kw = dict(batch_size=1, collate_fn=first, num_workers=args.num_workers,
                 persistent_workers=args.num_workers > 0)
    aug = None
    if args.aug_flip > 0 or args.aug_portrait > 0:
        centers = subject_centers(args.spatial_src)
        aug = {"flip": args.aug_flip, "portrait": args.aug_portrait, "centers": centers}
        print("augmentation: flip %.2f, portrait %.2f (%d clips with a labeled subject)"
              % (args.aug_flip, args.aug_portrait, len(centers)))
    train_ds = SFTRows(train_rows, processor, args.nframes, aug)
    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = (torch.utils.data.DataLoader(SFTRows(val_rows, processor, args.nframes),
                                          shuffle=False, **dl_kw) if val_rows else None)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_rows) / args.grad_accum) * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    print("optimizer steps: %d (warmup %d)" % (total_steps, warmup))

    def save(tag):
        if args.no_save:
            return
        d = os.path.join(args.out, tag)
        model.save_pretrained(d)
        print("  saved adapter -> %s" % d)

    step, acc_loss, acc_n = 0, 0.0, 0
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def optimizer_step(ep, n_micro):
        nonlocal step, acc_loss, acc_n
        if n_micro < args.grad_accum:
            # Partial accumulation at the end of an epoch: the losses were
            # divided by grad_accum, rescale to the mean over n_micro samples.
            # 轮末不足一整批：按实际样本数重缩放成均值。
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(args.grad_accum / float(n_micro))
        lr = lr_at(step, total_steps, args.lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        dt = time.time() - t0
        rec = {"epoch": ep, "step": step, "loss": round(acc_loss / acc_n, 4),
               "lr": lr, "grad_norm": round(gn, 4),
               "sec_per_step": round(dt / step, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log(rec)
        print("  ep %d step %d/%d loss %.4f lr %.2e gn %.3f %.1fs/step peak %.2fGB"
              % (ep, step, total_steps, rec["loss"], lr, gn,
                 rec["sec_per_step"], rec["peak_gb"]), flush=True)
        acc_loss, acc_n = 0.0, 0

    try:
        for ep in range(1, args.epochs + 1):
            # Counted per epoch, so a partial accumulation never leaks into the
            # next epoch and the last one of the run is applied, not dropped.
            # 按轮计数：轮末残余梯度当轮用掉，不带进下一轮。
            micro = 0
            for inp in train_dl:
                if "error" in inp:
                    print("  [skip] %s" % inp["error"])
                    log({"skip": inp["error"]})
                    continue
                loss = sample_loss(model, inp)
                (loss / args.grad_accum).backward()
                acc_loss += loss.item()
                acc_n += 1
                micro += 1
                if micro % args.grad_accum == 0:
                    optimizer_step(ep, args.grad_accum)
            if micro % args.grad_accum:
                optimizer_step(ep, micro % args.grad_accum)

            if val_dl is not None:
                vl, vn = evaluate(model, val_dl)
                log({"epoch": ep, "step": step, "val_loss": round(vl, 4), "val_n": vn})
                print("== epoch %d val_loss %.4f (%d samples)" % (ep, vl, vn), flush=True)
            save("epoch%d" % ep)
    except KeyboardInterrupt:
        print("interrupted at step %d" % step)
        save("interrupted")
    finally:
        log_f.close()
    print("done in %.1f min" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
