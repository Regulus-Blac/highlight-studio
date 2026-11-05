#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the temporal-only SFT dataset / 构建时序 SFT 数据集.

Turns the rebuilt weak-label rows (`data/train.with_video.jsonl`, produced by
`build_train_clips.py`) into chat-format rows a video-LLM can be fine-tuned on.
Only the TEMPORAL task is covered here -- which frames belong in the cut. The
spatial labels in this file are too weak to train on (x has stdev 0.106 and is
exactly 0.5 on a quarter of the keyframes, and 37% of rows have none at all),
so teaching the crop position from them mostly reproduces a center crop, which
costs nothing to compute directly.
只做时序：空间标注太弱，学出来约等于居中裁剪。

    python3 tools/build_sft_temporal.py \
        --input data/train.with_video.jsonl \
        --out-dir data/sft

Emits `sft_temporal_train.jsonl` and `sft_temporal_val.jsonl`, keeping the
`dataset_split` already recorded in the source rows.

Target format is the same JSON the shipped baseline already parses:

    {"segments": [[start_sec, end_sec], ...]}

Seconds, not frame indices: source fps ranges over 23.98/24/25/29.97/30, so a
frame index means nothing without the file, while a timestamp is stable. The
evaluator converts back to frames with the real per-video fps.
用秒而非帧号：fps 有五种，帧号脱离文件没有意义。

PROMPT must stay byte-identical between training and inference. It is defined
once here; `--dump-prompt` prints it so the inference side can be checked
against it. 训练与推理的提示词必须逐字一致。

`--task scores` builds the per-second variant instead: the video is split into
one-second steps (two sampled frames each = one Qwen3.5 temporal patch, which
the processor stamps with `<t seconds>`), and the target is one digit per step,
digit = round(9 x fraction of that second inside a highlight). Inference reads
the digit distribution at every step as a probability and picks frames with
`decode_scores.py`. 逐秒打分：每秒一个数字 0-9，= 该秒被高光覆盖的比例。

    python3 tools/build_sft_temporal.py --task scores \
        --input data/train.with_video.jsonl --out-dir data/sft
"""

import os
import re
import sys
import math
import json
import argparse
import collections
import statistics as st

# Target JSON the model must emit. Keep in sync with the inference prompt.
PROMPT = (
    "This is a {duration:.1f}-second video. Find every clip that is worth "
    "keeping in a highlight cut -- the moments a viewer would want to re-watch "
    "-- and give each one as a time interval in seconds, counted from the start "
    "of the video.\n"
    "Keep only genuinely highlight-worthy spans; do not return the whole video. "
    "If nothing qualifies, return an empty list.\n"
    "Output ONLY JSON, no extra text, strictly: "
    "{{\"segments\": [[start_sec, end_sec], ...]}}"
)


def render_prompt(duration):
    return PROMPT.format(duration=duration)


def render_target(segments, nd=2):
    """Segments -> the exact assistant string. 生成目标文本。"""
    segs = [[round(float(s["start_sec"]), nd), round(float(s["end_sec"]), nd)]
            for s in segments]
    segs.sort(key=lambda ab: ab[0])
    return json.dumps({"segments": segs}, ensure_ascii=False)


# Per-second scoring task. Keep in sync with tools/infer_scores.py (it imports
# these). 逐秒打分任务的提示词与标签，推理侧直接 import。
SCORE_PROMPT = (
    "This is a {duration:.1f}-second video shown as {k} one-second steps, each "
    "marked with its timestamp. For every step, in order, rate how much of it "
    "belongs in a highlight cut -- the moments a viewer would want to re-watch: "
    "0 = not a highlight, 9 = entirely highlight.\n"
    "Output ONLY {k} digits, one per step, with no spaces or other text."
)
# A tail shorter than this joins the previous step instead of becoming a step
# of its own. 末尾不足 0.25 s 的零头并入最后一步。
LAST_STEP_MIN_SEC = 0.25


def score_steps(duration):
    """Number of one-second steps for a clip of `duration` seconds."""
    return max(1, int(math.ceil(duration - LAST_STEP_MIN_SEC)))


def step_bounds(k, n_steps, duration):
    """[start, end) seconds of step k; the last step runs to the clip end."""
    return float(k), (float(duration) if k == n_steps - 1 else float(k + 1))


def render_score_prompt(duration):
    return SCORE_PROMPT.format(duration=duration, k=score_steps(duration))


def score_digits(segments, duration):
    """Segments -> one digit per step: round(9 x covered fraction). 逐秒覆盖率。"""
    n = score_steps(duration)
    spans = [(float(s["start_sec"]), float(s["end_sec"])) for s in segments]
    out = []
    for k in range(n):
        a, b = step_bounds(k, n, duration)
        cov = sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)
        out.append(int(min(1.0, cov / (b - a)) * 9 + 0.5))
    return out


def render_score_target(segments, duration):
    return "".join(str(d) for d in score_digits(segments, duration))


def parse_scores(text):
    """First run of digits (spaces / commas between them tolerated) -> [int]."""
    m = re.search(r"\d(?:[\s,]*\d)*", text)
    return [int(c) for c in re.sub(r"\D", "", m.group(0))] if m else []


def parse_segments_sec(text):
    """Mirror of the parser in baseline/baseline_qwen.py (kept dependency-free)."""
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        segs = obj.get("segments")
        if isinstance(segs, list):
            out = []
            for s in segs:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    try:
                        out.append((float(s[0]), float(s[1])))
                    except (TypeError, ValueError):
                        continue
            return out
    return []


def main():
    ap = argparse.ArgumentParser(description="Build temporal SFT data")
    ap.add_argument("--input", default="data/train.with_video.jsonl")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--task", choices=("segments", "scores"), default="segments",
                    help="segments: [[start,end],...] JSON; scores: one digit per second")
    ap.add_argument("--prefix", default=None,
                    help="output prefix (default sft_temporal / sft_scores)")
    ap.add_argument("--min-quality", type=float, default=0.0,
                    help="drop rows whose quality.score is below this")
    ap.add_argument("--max-hl-ratio", type=float, default=1.0,
                    help="drop rows whose highlight covers more than this "
                         "fraction of the clip (near-1.0 rows teach 'keep "
                         "everything', which collapses precision)")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--dump-prompt", action="store_true",
                    help="print the prompt template and exit")
    args = ap.parse_args()

    scores = args.task == "scores"
    prompt_fn = render_score_prompt if scores else render_prompt
    prefix = args.prefix or ("sft_scores" if scores else "sft_temporal")
    if args.dump_prompt:
        print(prompt_fn(12.3))
        return 0

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print("  input rows          : %d" % len(rows))

    os.makedirs(args.out_dir, exist_ok=True)
    buckets = collections.defaultdict(list)
    stats = collections.Counter()
    nsegs, ratios, durs, tgt_len = [], [], [], []
    digit_hist = collections.Counter()

    for d in rows:
        vp = d.get("video_path") or ""
        if not vp or not os.path.exists(vp):
            stats["dropped: no video file"] += 1
            continue
        segs = d.get("segments") or []
        if not segs:
            stats["dropped: no segments"] += 1
            continue
        q = (d.get("quality") or {}).get("score")
        if q is not None and q < args.min_quality:
            stats["dropped: quality below threshold"] += 1
            continue

        clip = d["clip"]
        duration = float(clip["end_sec"]) - float(clip["start_sec"])
        if duration <= 0:
            stats["dropped: non-positive duration"] += 1
            continue

        covered = sum(float(s["end_sec"]) - float(s["start_sec"]) for s in segs)
        hl_ratio = covered / duration
        if hl_ratio > args.max_hl_ratio:
            stats["dropped: highlight ratio too high"] += 1
            continue

        # Every segment must sit inside the clip, or the label is unlearnable.
        # 标签必须落在片段时间范围内。
        if any(float(s["start_sec"]) < -1e-6
               or float(s["end_sec"]) > duration + 1e-3
               or float(s["end_sec"]) <= float(s["start_sec"]) for s in segs):
            stats["dropped: segment outside the clip"] += 1
            continue

        # The rendered target must survive the inference-side parser unchanged,
        # otherwise training and evaluation disagree about what the model said.
        # 目标文本必须能被推理侧解析器原样读回。
        if scores:
            digits = score_digits(segs, duration)
            target = render_score_target(segs, duration)
            ok = parse_scores(target) == digits
        else:
            target = render_target(segs, args.decimals)
            want = [(round(float(s["start_sec"]), args.decimals),
                     round(float(s["end_sec"]), args.decimals)) for s in segs]
            ok = sorted(parse_segments_sec(target)) == sorted(want)
        if not ok:
            stats["dropped: target failed the round-trip check"] += 1
            continue

        rec = {
            "video_id": d["video_id"],
            "video": vp,
            "duration": round(duration, 3),
            "targetRatioWH": d["targetRatioWH"],
            "task": args.task,
            "messages": [
                {"role": "user", "content": [
                    {"type": "video", "video": vp},
                    {"type": "text", "text": prompt_fn(duration)},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": target},
                ]},
            ],
            "meta": {
                "n_segments": len(segs),
                "hl_ratio": round(hl_ratio, 4),
                "quality_score": q,
                "source_group": d["provenance"]["source_group"],
                "has_spatial": bool(d.get("cropRois")),
            },
        }
        buckets[d.get("dataset_split") or "train"].append(rec)
        stats["kept"] += 1
        nsegs.append(len(segs))
        ratios.append(hl_ratio)
        durs.append(duration)
        tgt_len.append(len(target))
        if scores:
            digit_hist.update(digits)

    for split, recs in sorted(buckets.items()):
        path = os.path.join(args.out_dir, "%s_%s.jsonl" % (prefix, split))
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  %-20s %5d rows -> %s" % (split, len(recs), path))

    print("=" * 68)
    for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("  %-38s %d" % (k, v))
    if nsegs:
        q = lambda a: (round(min(a), 2), round(st.median(a), 2), round(max(a), 2))
        print("-" * 68)
        print("  segments per clip  min/med/max : %s" % (q(nsegs),))
        print("  highlight ratio    min/med/max : %s" % (q(ratios),))
        print("  clip duration (s)  min/med/max : %s" % (q(durs),))
        print("  target chars       min/med/max : %s" % (q(tgt_len),))
        print("  n_segments histogram           : %s"
              % dict(sorted(collections.Counter(nsegs).items())))
        if scores:
            print("  steps per clip     min/med/max : %s"
                  % (q([score_steps(x) for x in durs]),))
            tot = sum(digit_hist.values())
            print("  digit histogram (%%)            : %s"
                  % {k: round(100.0 * v / tot, 1) for k, v in sorted(digit_hist.items())})
    print("-" * 68)
    print("  prompt (same string must be used at inference):")
    for line in prompt_fn(12.3).splitlines():
        print("    | %s" % line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
